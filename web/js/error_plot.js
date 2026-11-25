export class ErrorPlot {
 constructor(canvas){this.canvas=canvas;this.ctx=canvas.getContext('2d');this.samples=[];this.window=12;this.width=canvas.clientWidth||300;this.height=canvas.clientHeight||100;this.dpr=1;this.dirty=true;this.lastDraw=-Infinity;
 this.resize=new ResizeObserver(entries=>{const {width,height}=entries[0].contentRect;this.width=width;this.height=height;this.dirty=true;});this.resize.observe(canvas);}
 renderIfNeeded(now){const dpr=Math.min(window.devicePixelRatio||1,1.5);if((!this.dirty&&dpr===this.dpr)||now-this.lastDraw<66)return;this.dpr=dpr;this.lastDraw=now;this.dirty=false;this.draw();}
 push(time,value){const last=this.samples.at(-1);if(last&&time<last.time)this.samples=[];if(last&&time===last.time)return;this.samples.push({time,value});while(this.samples.length&&this.samples[0].time<time-this.window)this.samples.shift();this.dirty=true;}
 clear(){this.samples=[];this.dirty=true;}
 draw(){const c=this.canvas,ctx=this.ctx,{width:w,height:h,dpr}=this;if(!w||!h)return;if(c.width!==Math.round(w*dpr)||c.height!==Math.round(h*dpr)){c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 const s=this.samples,max=s.length?Math.max(...s.map(p=>p.value)):0,mean=s.length?s.reduce((a,p)=>a+p.value,0)/s.length:0;
 document.getElementById('error-mean').textContent=mean.toFixed(1)+' mm';document.getElementById('error-max').textContent=max.toFixed(1)+' mm';
 const top=10,bottom=h-20,left=26,right=w-4,ymax=Math.max(10,Math.ceil(max/10)*10),end=s.at(-1)?.time||0;
 ctx.font='8px ui-monospace, monospace';ctx.lineWidth=1;for(let i=0;i<=3;i++){const y=top+(bottom-top)*i/3;ctx.strokeStyle='#eaf0e9';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillStyle='#97a699';ctx.fillText((ymax*(1-i/3)).toFixed(0),0,y+3);}for(let i=0;i<=4;i++){const x=left+(right-left)*i/4;ctx.strokeStyle='#edf2ec';ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.stroke();ctx.fillStyle='#97a699';ctx.textAlign=i===4?'right':'left';ctx.fillText(i===4?'now':`−${12-i*3}s`,x,bottom+15);}ctx.textAlign='left';if(!s.length)return;
 const pts=s.map(p=>[left+(p.time-end+this.window)/this.window*(right-left),bottom-p.value/ymax*(bottom-top)]);
 const path=()=>{ctx.beginPath();ctx.moveTo(...pts[0]);for(let i=1;i<pts.length;i++){ctx.lineTo(...pts[i]);}};
 path();ctx.lineTo(pts.at(-1)[0],bottom);ctx.lineTo(pts[0][0],bottom);ctx.closePath();const g=ctx.createLinearGradient(0,top,0,bottom);g.addColorStop(0,'#8aaf7940');g.addColorStop(1,'#8aaf7904');ctx.fillStyle=g;ctx.fill();path();ctx.strokeStyle='#648d52';ctx.lineWidth=1.7;ctx.stroke();ctx.beginPath();ctx.arc(...pts.at(-1),2.5,0,Math.PI*2);ctx.fillStyle='#648d52';ctx.fill();
 }
}
