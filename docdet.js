/* docdet.js - nhận 4 góc tờ giấy bằng AI (DocAligner, Apache-2.0) chạy hoàn toàn trên máy.
   Không gửi ảnh đi đâu. Lỗi bất kỳ -> trả null để app quay về thuật toán cũ. */
(function(){
'use strict';
const DD = window.DocDet = { ready:false, loading:null, failed:false, base:'./', log:()=>{} };
let ort=null; const sess={}; const SIZES={S:'vendor/models/sa24d_q8.onnx.gz', L:'vendor/models/lcnet100_q8.onnx.gz', T:'vendor/models/t8_q8.onnx.gz'};

async function gunzip(buf){
  const h=new Uint8Array(buf,0,2); if(!(h[0]===0x1f&&h[1]===0x8b)) return buf;   // máy chủ đã tự giải nén sẵn
  const s=new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
  return await new Response(s).arrayBuffer();
}
async function getBuf(url){ const r=await fetch(url); if(!r.ok) throw new Error('tải lỗi '+url); return await r.arrayBuffer(); }
function loadScript(src){ return new Promise((ok,no)=>{ const s=document.createElement('script'); s.src=src; s.onload=ok; s.onerror=()=>no(new Error('script '+src)); document.head.appendChild(s); }); }

DD.init = function(opts){
  if(DD.loading) return DD.loading;
  opts=opts||{}; if(opts.base!==undefined) DD.base=opts.base;
  const models = opts.models||['S','L'];
  DD.loading=(async()=>{
    if(typeof DecompressionStream==='undefined') throw new Error('trình duyệt chưa hỗ trợ giải nén');
    await loadScript(DD.base+'vendor/ort/ort.wasm.min.js');
    ort=window.ort; ort.env.wasm.numThreads=1; ort.env.wasm.simd=true;
    if(opts.proxy!==false) ort.env.wasm.proxy=true;   // chạy mô hình trong luồng nền, không làm đơ giao diện
    ort.env.wasm.wasmPaths=new URL(DD.base+'vendor/ort/',document.baseURI).href;
    ort.env.wasm.wasmBinary=await gunzip(await getBuf(DD.base+'vendor/ort/ort-wasm-simd-threaded.wasm.gz'));
    // mô hình đầu tiên (nhẹ) xong là dùng được ngay; mô hình mạnh tải nền tiếp
    let first=true;
    for(const k of models){
      try{
        const buf=await gunzip(await getBuf(DD.base+SIZES[k]));
        sess[k]=await ort.InferenceSession.create(new Uint8Array(buf),{executionProviders:['wasm'],graphOptimizationLevel:'all'});
        DD.log('mô hình '+k+' sẵn sàng');
        if(first){ DD.ready=true; first=false; }
        if(opts.onModel) opts.onModel(k);
      }catch(e){ if(first) throw e; console.warn('DocDet: mô hình '+k+' lỗi',e); DD.partial=true; }
    }
    return true;
  })().catch(e=>{ DD.failed=true; DD.error=e; console.warn('DocDet:',e); return false; });
  return DD.loading;
};
DD.has = k=>!!sess[k];

/* ---------- chạy mô hình, lấy 4 góc (TL,TR,BR,BL) ---------- */
let inC=null, inX=null;
async function runModel(k, src, size){
  const IN=(k==='S'&&size)?size:256;   // chỉ mô hình mạnh (S) nhận cỡ nhỏ hơn; mô hình nhẹ cố định 256
  const SW=src.videoWidth||src.width, SH=src.videoHeight||src.height;
  if(!inC) inC=document.createElement('canvas');
  if(inC.width!==IN){ inC.width=IN; inC.height=IN; inX=inC.getContext('2d',{willReadFrequently:true}); }
  inX.imageSmoothingEnabled=true; inX.imageSmoothingQuality='low';   // mô hình học với kiểu thu nhỏ này (nội suy tuyến tính, không làm mịn)
  if(src.videoWidth){   // khung video: chép qua canvas trước, vì thu nhỏ trực tiếp từ video cho kết quả khác
    const k=Math.min(1,1280/Math.max(SW,SH)), vw=Math.round(SW*k), vh=Math.round(SH*k);
    if(!DD._vc){ DD._vc=document.createElement('canvas'); }
    if(DD._vc.width!==vw||DD._vc.height!==vh){ DD._vc.width=vw; DD._vc.height=vh; DD._vx=DD._vc.getContext('2d'); }
    DD._vx.drawImage(src,0,0,vw,vh); inX.drawImage(DD._vc,0,0,IN,IN);
  } else inX.drawImage(src,0,0,IN,IN);
  const px=inX.getImageData(0,0,IN,IN).data, n=IN*IN, f=new Float32Array(3*n);
  for(let i=0;i<n;i++){ f[i]=px[i*4+2]/255; f[n+i]=px[i*4+1]/255; f[2*n+i]=px[i*4]/255; }   // thứ tự BGR như DocAligner
  const out=await sess[k].run({img:new ort.Tensor('float32',f,[1,3,IN,IN])});
  const hm=out.heatmap.data, HW=IN/2, q=[], conf=[];
  for(let c=0;c<4;c++){
    const o=c*HW*HW; let bi=0,bv=-1;
    for(let i=0;i<HW*HW;i++){ const v=hm[o+i]; if(v>bv){bv=v;bi=i;} }
    const by=(bi/HW)|0, bx=bi%HW, r=6; let sw=0,sx=0,sy=0;
    for(let y=Math.max(0,by-r);y<=Math.min(HW-1,by+r);y++) for(let x=Math.max(0,bx-r);x<=Math.min(HW-1,bx+r);x++){
      const v=hm[o+y*HW+x]; if(v>=bv*0.3){ sw+=v; sx+=v*x; sy+=v*y; } }
    q.push([((sx/sw)+0.5)/HW*SW, ((sy/sw)+0.5)/HW*SH]); conf.push(bv);
  }
  q.conf=conf; return q;
}

/* ---------- kiểm tra khung bằng mép thật trong ảnh ---------- */
function boxBlur(a,w,h,r){
  const o=new Float32Array(a.length), t=new Float32Array(a.length);
  for(let y=0;y<h;y++){ let s=0,n=0; const row=y*w;
    for(let x=-r;x<w;x++){ const xa=x+r; if(xa<w){s+=a[row+xa];n++;} const xr=x-r-1; if(xr>=0){s-=a[row+xr];n--;} if(x>=0) t[row+x]=s/n; } }
  for(let x=0;x<w;x++){ let s=0,n=0;
    for(let y=-r;y<h;y++){ const ya=y+r; if(ya<h){s+=t[ya*w+x];n++;} const yr=y-r-1; if(yr>=0){s-=t[yr*w+x];n--;} if(y>=0) o[y*w+x]=s/n; } }
  return o;
}
let pC=null;
function prep(src,side){
  const SW=src.videoWidth||src.width, SH=src.videoHeight||src.height, s=Math.min(1,side/Math.max(SW,SH));
  const w=Math.max(8,Math.round(SW*s)), h=Math.max(8,Math.round(SH*s));
  if(!pC) pC=document.createElement('canvas'); pC.width=w; pC.height=h;
  const x=pC.getContext('2d',{willReadFrequently:true}); x.imageSmoothingQuality='high'; x.drawImage(src,0,0,w,h);
  const px=x.getImageData(0,0,w,h).data, N=w*h; let g=new Float32Array(N);
  for(let i=0;i<N;i++) g[i]=0.299*px[i*4]+0.587*px[i*4+1]+0.114*px[i*4+2];
  g=boxBlur(boxBlur(g,w,h,1),w,h,1);
  const gx=new Float32Array(N), gy=new Float32Array(N), mag=new Float32Array(N); let mm=0;
  for(let y=1;y<h-1;y++) for(let xx=1;xx<w-1;xx++){
    const i=y*w+xx, a=g[i-w-1],b=g[i-w],c=g[i-w+1],d=g[i-1],f=g[i+1],e=g[i+w-1],hh=g[i+w],k=g[i+w+1];
    const X=(c+2*f+k)-(a+2*d+e), Y=(e+2*hh+k)-(a+2*b+c); gx[i]=X; gy[i]=Y; const m=Math.hypot(X,Y); mag[i]=m; if(m>mm) mm=m; }
  const BINS=512, hist=new Uint32Array(BINS), sc=(BINS-1)/(mm||1);
  for(let i=0;i<N;i++) hist[(mag[i]*sc)|0]++;
  let acc=0, bi=0; const tg=N*0.88; for(;bi<BINS;bi++){ acc+=hist[bi]; if(acc>=tg) break; }
  return {s,w,h,g,gx,gy,mag,thr:Math.max(24,bi/sc)};
}
const dist=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
function edgeSupport(q,P){
  const res=[], {w,h,mag,gx,gy,thr}=P;
  for(let e=0;e<4;e++){
    const a=[q[e][0]*P.s,q[e][1]*P.s], b=[q[(e+1)%4][0]*P.s,q[(e+1)%4][1]*P.s], L=dist(a,b);
    if(L<6){ res.push(0); continue; }
    const nx=-(b[1]-a[1])/L, ny=(b[0]-a[0])/L, n=Math.max(14,Math.floor(L/3)); let hit=0,tot=0;
    for(let j=0;j<n;j++){
      const t=0.06+0.88*j/(n-1), x0=a[0]+(b[0]-a[0])*t, y0=a[1]+(b[1]-a[1])*t;
      if(x0<2||y0<2||x0>w-3||y0>h-3) continue; tot++;
      for(let s2=-3;s2<=3;s2++){
        const xi=Math.round(x0+nx*s2), yi=Math.round(y0+ny*s2); if(xi<1||yi<1||xi>=w-1||yi>=h-1) continue;
        const i=yi*w+xi, m=mag[i];
        if(m>thr*0.7 && Math.abs(gx[i]*nx+gy[i]*ny)>0.75*m){ hit++; break; }
      }
    }
    res.push(tot?hit/tot:0);
  }
  return res;
}
function inter(p1,d1,p2,d2){
  const det=d1[0]*(-d2[1])-(-d2[0])*d1[1]; if(Math.abs(det)<1e-9) return null;
  const rx=p2[0]-p1[0], ry=p2[1]-p1[1], t=(rx*(-d2[1])-(-d2[0])*ry)/det;
  return [p1[0]+t*d1[0], p1[1]+t*d1[1]];
}
function angBetween(a,b){ let d=Math.abs(Math.atan2(a[1],a[0])-Math.atan2(b[1],b[0]))*180/Math.PI%180; return Math.min(d,180-d); }
function crease(P,p,d){   // đường gáy: tối hơn giấy 2 bên
  const {w,h,g}=P; const nx=-d[1], ny=d[0];
  const lm=(ox,oy)=>{ let s=0,n=0; for(let k=0;k<40;k++){ const t=-0.42+0.84*k/39, xi=Math.round(ox+d[0]*t), yi=Math.round(oy+d[1]*t); if(xi>=0&&yi>=0&&xi<w&&yi<h){ s+=g[yi*w+xi]; n++; } } return n>20?s/n:NaN; };
  const a=lm(p[0]+nx*5,p[1]+ny*5), b=lm(p[0]-nx*5,p[1]-ny*5), c=lm(p[0],p[1]);
  return Math.min(a,b)-c;
}
/* một cạnh yếu + lệch hướng cạnh đối diện >10° (hay gặp ở cạnh gáy): dựng lại cạnh song song cạnh đối diện */
function repairEdge(q,sup,P){
  for(let e=0;e<4;e++){
    const a=q[e], b=q[(e+1)%4], o1=q[(e+2)%4], o0=q[(e+3)%4], opp=[o1[0]-o0[0],o1[1]-o0[1]];
    if(sup[e]>=0.5 || angBetween([b[0]-a[0],b[1]-a[1]],opp)<10) continue;
    const L=Math.hypot(opp[0],opp[1]), d=[opp[0]/L,opp[1]/L]; let best=null;
    const opts=[{kp:a, ne:(e+1)%4, mv:(e+1)%4},{kp:b, ne:(e+3)%4, mv:e}];
    for(const o of opts){
      const p1=q[o.ne], p2=q[(o.ne+1)%4], X=inter(o.kp,d,p1,[p2[0]-p1[0],p2[1]-p1[1]]); if(!X) continue;
      const sc=crease(P,[o.kp[0]*P.s,o.kp[1]*P.s],d);
      if(!best||sc>best.sc) best={sc,mv:o.mv,X};
    }
    if(best){ const nq=q.map(p=>p.slice()); nq[best.mv]=best.X; nq.split=e; return nq; }
  }
  return null;
}
const minOf=a=>Math.min.apply(null,a);
const valid=(q,SW,SH)=>{
  if(!q||q.some(p=>!isFinite(p[0])||!isFinite(p[1]))) return false;
  let s=0, sg=0; for(let i=0;i<4;i++){ const a=q[i],b=q[(i+1)%4],c=q[(i+2)%4]; s+=a[0]*b[1]-b[0]*a[1];
    const cr=(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]), z=Math.sign(cr); if(!z) return false; if(!sg) sg=z; else if(z!==sg) return false; }
  const ar=Math.abs(s)/2; return ar>SW*SH*0.08;
};

DD._i={runModel,prep,edgeSupport,valid};
/* ---------- hàm chính ----------
   opts.fast: chỉ dùng mô hình nhẹ (xem trước camera). Mặc định: mô hình mạnh + kiểm tra + sửa cạnh gáy. */
DD.detect = async function(src,opts){
  opts=opts||{};
  if(!DD.ready) return null;
  try{
    const SW=src.videoWidth||src.width, SH=src.videoHeight||src.height, t0=performance.now();
    const P=prep(src,opts.edgeSide||480); const info={};
    const fastK=sess.T?'T':'L';
    const order=(opts.fast||!sess.S)?[fastK]:['S',fastK];
    let best=null;
    for(const k of order){
      if(!sess[k]) continue;
      const q=await runModel(k,src,opts.size); if(!valid(q,SW,SH)) continue;
      const sup=edgeSupport(q,P), mn=minOf(sup);
      info[k]={sup,mn};
      const weak=[]; sup.forEach((v,i)=>{ if(v<0.5) weak.push(i); });
      if(!weak.length){ best={q,k,sup}; break; }
      if(weak.length===1){
        const e=weak[0], a=q[e], b=q[(e+1)%4], o1=q[(e+2)%4], o0=q[(e+3)%4];
        if(angBetween([b[0]-a[0],b[1]-a[1]],[o1[0]-o0[0],o1[1]-o0[1]])<10){ q.split=e; best={q,k:k+'g',sup}; break; }   // cạnh gáy: yếu mép nhưng thẳng hàng
        const rq=repairEdge(q,sup,P);
        if(rq&&valid(rq,SW,SH)){ best={q:rq,k:k+'+sửa',sup:edgeSupport(rq,P)}; break; }
      }
      if(!best||mn>best.mn) best={q,k:k+'?',sup,mn};
    }
    if(!best) return null;
    const q=best.q.map(p=>[Math.min(SW,Math.max(0,p[0])),Math.min(SH,Math.max(0,p[1]))]);
    if(best.q.split!==undefined){ q.split=best.q.split; }
    q.ai=best.k; q.minSup=minOf(best.sup); q.ms=Math.round(performance.now()-t0); q.info=info;
    return q;
  }catch(e){ console.warn('DocDet.detect',e); return null; }
};
})();
