// AOT graph kernels. All memory is owned by the loaded engine, not by a call.
const f = Math.fround;

export function dense(s,w,src,dst,weight,bias,cols,rows) {
  const stop = cols - cols % 4;
  for (let i=0; i<rows; i++,weight+=cols) {
    let a=0,b=0,c=0,d=0,j=0;
    for (;j<stop;j+=4) {
      a+=s[src+j]*w[weight+j];
      b+=s[src+j+1]*w[weight+j+1];
      c+=s[src+j+2]*w[weight+j+2];
      d+=s[src+j+3]*w[weight+j+3];
    }
    for (;j<cols;j++) a+=s[src+j]*w[weight+j];
    s[dst+i]=(a+b+c+d)+w[bias+i];
  }
}

export function norm(s,w,src,dst,scale,bias,n,epsilon) {
  let sum=0,sq=0;
  for(let i=0;i<n;i++) { const v=s[src+i]; sum+=v; sq+=v*v; }
  const mean=f(sum/n), sd=f(Math.sqrt(Math.max(0,f(f(sq/n)-f(mean*mean)))+epsilon));
  for(let i=0;i<n;i++) s[dst+i]=f(f(f(s[src+i]-mean)/sd)*w[scale+i])+w[bias+i];
}

// Abramowitz–Stegun 7.1.26; maximum absolute error ~1.5e-7.
export function erf(x) {
  if(x===0)return x;
  const t=1/(1+.3275911*Math.abs(x));
  const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x);
  return x<0?-y:y;
}

export class Float32Engine {
  constructor(meta, forward, bytes) {
    if(meta.format!==1 || ![311,318].includes(meta.input) || meta.output!==6 ||
       !Number.isInteger(meta.scratch) || meta.scratch<=0 || !Number.isInteger(meta.floats) ||
       bytes.byteLength!==4*meta.floats || typeof forward!=='function') throw Error('Invalid JS policy artifact');
    this.width=meta.input;
    this.forward=forward;
    this.scratch=new Float32Array(meta.scratch);
    this.output=new Float32Array(meta.output);
    // Binary payload is explicitly little endian. Swap only at load if required.
    if(new Uint8Array(Uint32Array.of(1).buffer)[0]===1) {
      this.weights=new Float32Array(bytes.buffer,bytes.byteOffset,meta.floats);
    } else {
      const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
      this.weights=new Float32Array(meta.floats);
      for(let i=0;i<meta.floats;i++)this.weights[i]=view.getFloat32(4*i,true);
    }
    if(!this.weights.every(Number.isFinite))throw Error('Invalid JS policy weights');
  }
  // Borrowed result: consume/copy before the next call; no allocation on success.
  infer(obs) {
    if(!(obs instanceof Float32Array)||obs.length!==this.width)throw Error('Invalid policy input width');
    this.forward(obs,this.output,this.scratch,this.weights);
    for(let i=0;i<6;i++)if(!Number.isFinite(this.output[i]))throw Error('Nonfinite policy output');
    return this.output;
  }
}
