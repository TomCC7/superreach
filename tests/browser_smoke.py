"""Exercise the served worker, real WASM, and native Chromium controls."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import onnxruntime as ort
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8000')
parser.add_argument('--onnx', type=Path, required=True)
parser.add_argument('--chromium', type=Path)
parser.add_argument('--screenshot', type=Path)
args = parser.parse_args()
BASE = args.url.rstrip('/')
ID='tri-mode-dual-enc@13.5k'

def diff(a,b): return max(abs(x-y) for x,y in zip(a,b,strict=True))

with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path=args.chromium,headless=True,args=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000});errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
    page.on('response',lambda r:errors.append(f'{r.status} {r.url}') if r.status>=400 else None)
    page.goto(BASE+'/?v=trimode-v1');page.bring_to_front()
    wait=lambda e:page.wait_for_function(e,timeout=15000)
    js=page.evaluate
    wait(f'demo.state?.tick>8 && demo.state.checkpoint==={json.dumps(ID)} && demo.state.tri && demo.state.mode===0')
    js('''window.samples=[];demo.worker.addEventListener('message',({data})=>{if(data.type==='state')samples.push(data);});''')
    def switch(mode):
        before=js('demo.state.tick'); n=js('samples.length')
        page.locator('#control-mode').select_option(str(mode))
        wait(f'demo.state.mode==={mode} && demo.state.tick>{before+4} && demo.state.running && !document.querySelector("#control-mode").disabled')
        state=js(f'samples.slice({n}).find(s=>s.mode==={mode})')
        assert state['tick']>=before and state['running'] and state['preset']=='hold'
        if mode==2: assert diff(state['q'],state['qref'])<1e-9,state
        else: assert diff(state['tcp'],state['goal'])<1e-9,state
        return state['tick']
    t_xyz=switch(1)
    assert page.locator('#eef-tool').is_hidden() and page.locator('#joint-controls').is_hidden()
    # Native TransformControls raycast, then actual mouse press/move/release.
    js('''window.pickPoint=axis=>{
      const c=demo.gizmo;demo.scene.updateMatrixWorld(true);const r=demo.renderer.domElement.getBoundingClientRect(),v=c.object.position.clone().project(demo.camera);
      const xx=r.left+(v.x+1)*r.width/2,yy=r.top+(1-v.y)*r.height/2;
      for(let dx=-90;dx<=90;dx+=2)for(let dy=-90;dy<=90;dy+=2){
        const x=xx+dx,y=yy+dy;if(document.elementFromPoint(x,y)!==demo.renderer.domElement)continue;
        c.pointerHover({x:2*(x-r.left)/r.width-1,y:1-2*(y-r.top)/r.height});const hit=c.axis===axis;c.axis=null;if(hit)return [x,y];
      }
    };''')
    def drag(axis,dx,dy):
        point=js('axis=>pickPoint(axis)',axis);assert point,axis
        x,y=point;page.mouse.move(x,y);page.mouse.down();wait('demo.gizmo.dragging && !demo.orbit.enabled')
        page.mouse.move(x+dx,y+dy,steps=5);page.mouse.up();wait('!demo.gizmo.dragging && demo.orbit.enabled')
    old=js('demo.state.goal');drag('X',38,0)
    wait(f'Math.abs(demo.state.goal[0]-{old[0]})>.02');xyz_start=js('demo.state');xyz_goal=xyz_start['goal'][:3]
    xyz_error=math.dist(xyz_goal,xyz_start['tcp'][:3]);assert xyz_error>.01
    wait(f'demo.state.tick>{xyz_start["tick"]+85}')
    xyz_final=math.dist(xyz_goal,js('demo.state.tcp.slice(0,3)'));assert xyz_final<.65*xyz_error,(xyz_error,xyz_final)
    t_joint=switch(2)
    assert page.locator('#joint-controls input[type=range]').count()==6
    assert not js('demo.gizmo.enabled')
    slider=page.locator('#joint-1');slider.scroll_into_view_if_needed();r=slider.bounding_box();assert r
    start=js('demo.state.q[0]');page.mouse.click(r['x']+.60*r['width'],r['y']+.5*r['height'])
    wait(f'Math.abs(demo.state.qref[0]-{start})>.12');joint_start=js('demo.state');goal=joint_start['qref'][0]
    wait(f'demo.state.tick>{joint_start["tick"]+100}')
    final=js('demo.state.q[0]');assert abs(goal-final)<.45*abs(goal-start),(start,goal,final)
    slider.press('ArrowRight');wait(f'demo.state.qref[0]>{goal+0.001}')
    if args.screenshot:
        page.screenshot(path=args.screenshot)
    page.locator('#joint-home').click();wait('demo.state.qref[0]===0 && demo.state.qref[1]===1.047')
    t_eef=switch(0)
    page.locator('#eef-tool').select_option('rotate');wait('demo.gizmo.mode==="rotate"');page.locator('#viewport').scroll_into_view_if_needed()
    old=js('demo.state.goal');drag('Z',23,-35)
    wait(f'demo.state.goal.slice(3).some((x,i)=>Math.abs(x-({old[3:]})[i])>.02)')
    rotated=js('demo.state.goal');assert diff(rotated[:3],old[:3])<1e-9,(old,rotated)
    rows=js('demo.getTrace().then(x=>x.trace)');assert set(r['mode'] for r in rows)=={0,1,2}
    bundle=page.request.get(BASE+'/assets/BUNDLE.json').json();ent=bundle['checkpoints'][ID]
    weights=args.onnx.read_bytes();assert hashlib.sha256(weights).hexdigest()==ent['sha256']
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;cpu=ort.InferenceSession(weights,opts,providers=['CPUExecutionProvider'])
    maxima={i:0. for i in range(3)}
    for prev,row in zip([None]+rows,rows):
        mode=row['mode'];obs=np.array([row['obs']],np.float32);q=np.array(row['q_des']);assert obs.shape==(1,318) and obs[0,317]==mode
        target=np.asarray(row['goal_qref'] if mode==2 else [0]*6,np.float32)
        np.testing.assert_array_equal(obs[0,311:317],target)
        np.testing.assert_array_equal(obs[0,304:311],np.float32(row['goal']))
        if prev:
            assert row['tick']==prev['tick']+1
            np.testing.assert_array_equal(obs[0,285:304],np.float32(prev['q']+prev['dq']+prev['q_des']+[1]))
            np.testing.assert_array_equal(obs[0,:285],np.float32(prev['obs'][19:304]))
        assert np.isfinite(np.r_[q,row['q'],row['dq'],row['tcp']]).all()
        maxima[mode]=max(maxima[mode],float(np.abs(cpu.run(None,{'obs':obs})[0][0]-q).max()))
    assert max(maxima.values())<1e-5,maxima
    # Exercise every retained checkpoint through the real selector, then return to Group B.
    for cid in [k for k in bundle['checkpoints'] if k!=ID]+[ID]:
        page.locator('#checkpoint').select_option(cid)
        wait(f'demo.state.checkpoint==={json.dumps(cid)} && demo.state.tick>2 && !document.querySelector("#checkpoint").disabled')
        assert page.locator('#control-mode-card').is_visible()==(cid==ID)
    assert js('window.browserErrors.concat(demo.errors)')==[] and errors==[],errors
    data=dict(chromium=browser.version,frames=len(rows),browser_cpu_max_abs_rad=maxima,switch_ticks=[t_xyz,t_joint,t_eef],xyz_error_mm=[xyz_error*1000,xyz_final*1000],joint_deg=[math.degrees(x) for x in [start,goal,final]],errors=errors)
    print(json.dumps(data,indent=2));browser.close()
