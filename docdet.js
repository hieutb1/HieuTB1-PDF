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

/* ---------- dò sát mép SAU mô hình AI ----------
   Chỉ áp cho cạnh có mép giấy rõ (giấy sáng, nền tối bên ngoài) và không phải cạnh gáy.
   Mỗi cạnh dịch tối đa ~2% đường chéo; chỉ nhận khi tìm được mép thẳng rõ ràng, không thì giữ nguyên kết quả AI. */
let snapC=null;
function fitLineRobust(pts){
  let use=pts, f=null;
  for(let it=0;it<4;it++){
    let mx=0,my=0; for(const p of use){mx+=p[0];my+=p[1];} mx/=use.length; my/=use.length;
    let sxx=0,sxy=0,syy=0; for(const p of use){ const a=p[0]-mx,b=p[1]-my; sxx+=a*a; sxy+=a*b; syy+=b*b; }
    const th=0.5*Math.atan2(2*sxy,sxx-syy); f={x:mx,y:my,dx:Math.cos(th),dy:Math.sin(th)};
    const res=pts.map(p=>Math.abs((p[0]-mx)*f.dy-(p[1]-my)*f.dx));
    const sr=res.slice().sort((a,b)=>a-b), med=sr[sr.length>>1], lim=Math.max(1.2,2.5*med);
    const keep=pts.filter((p,i)=>res[i]<=lim); if(keep.length<8) break; use=keep;
  }
  const res=pts.map(p=>Math.abs((p[0]-f.x)*f.dy-(p[1]-f.y)*f.dx)), inl=[];
  res.forEach((r,i)=>{ if(r<=2) inl.push(r); });
  f.inl=inl.length; f.rms=inl.length?Math.sqrt(inl.reduce((a,b)=>a+b*b,0)/inl.length):99;
  return f;
}
function snapEdges(src,q,opts){
  opts=opts||{};
  const SW=src.videoWidth||src.width, SH=src.videoHeight||src.height;
  const s=Math.min(1,(opts.side||1400)/Math.max(SW,SH)), w=Math.round(SW*s), h=Math.round(SH*s), N=w*h;
  if(!snapC) snapC=document.createElement('canvas'); snapC.width=w; snapC.height=h;
  const x=snapC.getContext('2d',{willReadFrequently:true}); x.imageSmoothingQuality='high'; x.drawImage(src,0,0,w,h);
  const px=x.getImageData(0,0,w,h).data; let G=new Float32Array(N);
  for(let i=0;i<N;i++) G[i]=0.299*px[i*4]+0.587*px[i*4+1]+0.114*px[i*4+2];
  G=boxBlur(G,w,h,1);
  const at=(xx,yy)=>{ const xi=Math.round(xx), yi=Math.round(yy); if(xi<0||yi<0||xi>=w||yi>=h) return NaN; return G[yi*w+xi]; };
  const qs=q.map(p=>[p[0]*s,p[1]*s]), diag=Math.hypot(w,h), R=Math.max(6,Math.round(diag*(opts.maxShift||0.02)));
  const cx=(qs[0][0]+qs[1][0]+qs[2][0]+qs[3][0])/4, cy=(qs[0][1]+qs[1][1]+qs[2][1]+qs[3][1])/4;
  const paperL=(()=>{ const v=[]; for(let i=1;i<=5;i++) for(let j=1;j<=5;j++){ const u=i/6, k=j/6, top=[qs[0][0]+(qs[1][0]-qs[0][0])*u,qs[0][1]+(qs[1][1]-qs[0][1])*u], bot=[qs[3][0]+(qs[2][0]-qs[3][0])*u,qs[3][1]+(qs[2][1]-qs[3][1])*u]; const g=at(top[0]+(bot[0]-top[0])*k, top[1]+(bot[1]-top[1])*k); if(!isNaN(g)) v.push(g); } v.sort((a,b)=>a-b); return v.length?v[v.length>>1]:200; })();
  const lines=[], snapped=[], dbg={}, offsAll={};
  for(let e=0;e<4;e++){
    const a=qs[e], b=qs[(e+1)%4], len=dist(a,b), ux=(b[0]-a[0])/len, uy=(b[1]-a[1])/len;
    let nx=-uy, ny=ux; if(((a[0]+b[0])/2-cx)*nx+((a[1]+b[1])/2-cy)*ny<0){ nx=-nx; ny=-ny; }   // pháp tuyến hướng ra ngoài
    lines.push({x:a[0],y:a[1],dx:ux,dy:uy});
    if(q.split===e) continue;                                  // cạnh gáy: không dò
    // 1) mép rõ? giấy sáng bên trong, nền tối bên ngoài (đo bằng trung vị dọc cạnh)
    const D=Math.max(4,Math.round(R*0.6)), df=[], dfar=[], outN=[], outF=[];
    for(let j=0;j<24;j++){
      const t=0.1+0.8*j/23, x0=a[0]+(b[0]-a[0])*t, y0=a[1]+(b[1]-a[1])*t;
      const gi=at(x0-nx*D,y0-ny*D), go=at(x0+nx*D,y0+ny*D); if(isNaN(gi)||isNaN(go)) continue; df.push(gi-go); outN.push(go); const gf=at(x0+nx*D*2.6,y0+ny*D*2.6); if(!isNaN(gf)){ dfar.push(gi-gf); outF.push(gf); }
    }
    if(df.length<16) continue;
    const sd=df.slice().sort((p,r)=>p-r), med=sd[sd.length>>1];
    if(med<35 || df.filter(v=>v>=20).length<0.8*df.length){ dbg[e]='mep-yeu'; continue; }
    const md=a=>{ const z=a.slice().sort((p,r)=>p-r); return z[z.length>>1]; };
    if(md(outN)>125){ dbg[e]='nen-sang'; continue; }                     // nền bên ngoài phải tối
    if(dfar.length>=12 && (md(dfar)<30 || md(outF)>125)){ dbg[e]='ngoai-sang'; continue; }   // bên ngoài xa vẫn sáng (vd. trang bên cạnh) -> là gáy, không phải mép giấy
    // 2) vị trí chuyển sáng -> tối dọc pháp tuyến tại 48 điểm
    const M=48, W2=Math.max(3,Math.round(R*0.3)), pts=[], offs=[];
    for(let j=0;j<M;j++){
      const t=0.08+0.84*j/(M-1), x0=a[0]+(b[0]-a[0])*t, y0=a[1]+(b[1]-a[1])*t;
      let bs=-1e9, bi=null;
      for(let s2=-R;s2<=R;s2++){
        let si=0,so=0,ni=0,no=0;
        for(let u=1;u<=W2;u++){ const vi=at(x0+nx*(s2-u),y0+ny*(s2-u)), vo=at(x0+nx*(s2+u),y0+ny*(s2+u)); if(!isNaN(vi)){si+=vi;ni++;} if(!isNaN(vo)){so+=vo;no++;} }
        if(ni<2||no<2) continue;
        const sc=si/ni-so/no; if(sc>bs){bs=sc;bi=s2;}
      }
      if(bi!==null && bs>=30 && Math.abs(bi)<R){ pts.push([x0+nx*bi,y0+ny*bi]); offs.push(bi); }
    }
    offsAll[e]=offs.slice(); if(pts.length<0.7*M){ dbg[e]='it-diem'; continue; }
    // 3) kẻ đường thẳng, chỉ nhận khi thẳng rõ và gần hướng cũ
    const f=fitLineRobust(pts);
    if(f.inl<0.75*M || f.rms>1.6){ dbg[e]='khong-thang'; continue; }
    if(f.dx*ux+f.dy*uy<0){ f.dx=-f.dx; f.dy=-f.dy; }
    { const rs=pts.map(p=>(p[0]-f.x)*f.dy-(p[1]-f.y)*f.dx), k=Math.max(4,Math.round(pts.length*0.2)), m2=a=>{ const z=a.map(Math.abs).sort((p,r)=>p-r); return z[z.length>>1]; };
      if(m2(rs.slice(0,k))>2.5 || m2(rs.slice(-k))>2.5){ dbg[e]='cong'; continue; } }
    if(Math.abs(f.dx*uy-f.dy*ux)>Math.sin(3*Math.PI/180)) continue;          // lệch hướng quá 3°
    const mid=[(a[0]+b[0])/2,(a[1]+b[1])/2], sh=Math.abs((f.x-mid[0])*nx+(f.y-mid[1])*ny);
    if(Math.abs(((f.x-mid[0])*nx+(f.y-mid[1])*ny))>R*0.95) continue;
    void sh;
    {   // giấy phải thật sự nằm ngay bên trong đường mới (không phải vật sáng khác)
      let tmin=1e9,tmax=-1e9; for(const p of pts){ const tt=(p[0]-f.x)*f.dx+(p[1]-f.y)*f.dy; if(tt<tmin) tmin=tt; if(tt>tmax) tmax=tt; }
      let okc=0,tot2=0; for(let j=0;j<40;j++){ const tt=tmin+(tmax-tmin)*j/39, v=at(f.x+f.dx*tt-nx*4, f.y+f.dy*tt-ny*4); if(isNaN(v)) continue; tot2++; if(v>=0.7*paperL) okc++; }
      if(tot2<20 || okc<0.85*tot2){ dbg[e]='trong-toi'; continue; } }
    lines[e]=f; snapped.push(e);
  }
  // góc mới phải đúng là góc giấy: ngay bên trong (chéo vào) là giấy sáng, bên ngoài tối hơn rõ. Không đạt -> bỏ các cạnh đã dò kề góc đó.
  const cornerOK=(X,fac)=>{
    let ix=cx-X[0], iy=cy-X[1]; const L=Math.hypot(ix,iy)||1; ix/=L; iy/=L;
    const gi=at(X[0]+ix*6,X[1]+iy*6), go=at(X[0]-ix*6,X[1]-iy*6); if(isNaN(gi)) return false;
    return gi>=fac*paperL;
  };
  let out;
  for(let pass=0;pass<3;pass++){
    let bad=false;
    out=qs.map((p,i)=>{
      const e1=(i+3)%4, e2=i; if(snapped.indexOf(e1)<0 && snapped.indexOf(e2)<0) return p;
      const l1=lines[e1], l2=lines[e2], X=inter([l1.x,l1.y], [l1.dx,l1.dy], [l2.x,l2.y], [l2.dx,l2.dy]);
      const both=snapped.indexOf(e1)>=0 && snapped.indexOf(e2)>=0;   // kiểm tra bằng độ sáng chỉ với góc ngoài; góc sát gáy có bóng gáy nên bỏ qua
      const ok=X && dist(X,p)<=diag*0.03 && (both ? cornerOK(X,0.7) : dist(X,p)<=diag*0.007);   // góc sát gáy: bóng gáy làm tối nên chỉ cho dịch nhỏ
      if(!ok){ [e1,e2].forEach(e=>{ const k=snapped.indexOf(e); if(k>=0){ snapped.splice(k,1); dbg[e]='goc-sai'; lines[e]={x:qs[e][0],y:qs[e][1],dx:(qs[(e+1)%4][0]-qs[e][0])/dist(qs[e],qs[(e+1)%4]),dy:(qs[(e+1)%4][1]-qs[e][1])/dist(qs[e],qs[(e+1)%4])}; } }); bad=true; return p; }
      return [Math.min(w,Math.max(0,X[0])),Math.min(h,Math.max(0,X[1]))];
    });
    if(!bad) break;
  }
  const r=out.map(p=>[p[0]/s,p[1]/s]); r.snapped=snapped; r.dbg=dbg; r.offs=offsAll; return r;
}

/* ---------- làm mượt khung xem trước ----------
   - lấy trung bình vài lần dò gần nhất (các lần cùng chỉ về một chỗ)
   - bỏ qua lần dò kém tin cậy (giữ khung cũ)
   - lần dò lệch hẳn khung đang hiện: chỉ đổi khi 2 lần dò liên tiếp cùng chỉ về vị trí mới
   - agree = số lần dò liên tiếp khớp nhau (dùng để khoá khung / tự chụp) */
DD.makeSmoother=function(o){
  o=o||{}; const NEAR=o.near||0.03, SAME=o.same||0.02, DEAD=o.dead||0.004, KEEP=o.keep||3;
  const S={shown:null,hist:[],cand:null,candN:0,agree:0};
  const diff=(a,b,diag)=>Math.max.apply(null,a.map((p,i)=>dist(p,b[i])))/diag;
  const align=(q,ref)=>{ let best=q,bd=1e18; for(let k=0;k<4;k++){ const r=[0,1,2,3].map(i=>q[(i+k)%4]); const d=r.reduce((s,p,i)=>s+dist(p,ref[i]),0); if(d<bd){bd=d;best=r;} } return best; };
  const avg=l=>[0,1,2,3].map(i=>[l.reduce((s,q)=>s+q[i][0],0)/l.length, l.reduce((s,q)=>s+q[i][1],0)/l.length]);
  const lowConf=q=>/\?/.test(q.ai||'') || /^L/.test(q.ai||'') && o.strongLoaded || (q.meanConf!==undefined&&q.meanConf<0.25);
  S.reset=()=>{ S.shown=null; S.hist=[]; S.cand=null; S.candN=0; S.agree=0; };
  S.update=function(q,diag){
    const r={quad:S.shown, agree:S.agree, switched:false, ignored:false};
    if(!q) return r;
    const low=lowConf(q); const sp=q.split; q=q.map(p=>p.slice()); if(sp!==undefined) q.split=sp;
    if(!S.shown){
      if(low){ r.ignored=true; return r; }
      S.shown=q; S.hist=[q]; S.agree=0; r.quad=q; r.switched=true; return r;
    }
    if(low){ r.ignored=true; return r; }                   // kém tin cậy: giữ khung cũ
    q=align(q,S.shown); const d=diff(q,S.shown,diag);
    if(d<NEAR){
      S.cand=null; S.candN=0;
      const prev=S.hist[S.hist.length-1], dp=diff(q,prev,diag);
      if(dp<SAME) S.agree++; else S.agree=0;
      S.hist.push(q); if(S.hist.length>KEEP) S.hist.shift();
      const a=avg(S.hist); if(diff(a,S.shown,diag)>DEAD){ a.split=S.shown.split; S.shown=a; }
    } else {
      // lệch hẳn: giữ khung cũ, chờ lần dò thứ 2 cùng chỉ về chỗ mới
      if(S.cand && diff(align(q,S.cand),S.cand,diag)<NEAR){ S.candN++; S.cand=avg([S.cand,align(q,S.cand)]); } else { S.cand=q; S.candN=1; }
      r.ignored=true;
      if(S.candN>=2){ S.shown=S.cand; S.hist=[S.cand]; S.cand=null; S.candN=0; S.agree=0; r.switched=true; r.ignored=false; }
    }
    r.quad=S.shown; r.agree=S.agree; return r;
  };
  return S;
};
DD._i={runModel,prep,edgeSupport,valid,snapEdges};
/* ---------- hàm chính ----------
   opts.fast: chỉ dùng mô hình nhẹ (xem trước camera). Mặc định: mô hình mạnh + kiểm tra + sửa cạnh gáy. */
async function core(src,opts){
  opts=opts||{};
  if(!DD.ready) return null;
  try{
    const SW=src.videoWidth||src.width, SH=src.videoHeight||src.height, t0=performance.now();
    const P=prep(src,opts.edgeSide||480); const info={};
    const fastK=sess.T?'T':'L';
    // thử lần lượt: mô hình mạnh ở cỡ yêu cầu, rồi các cỡ khác (kết quả đổi theo cỡ nên một cỡ hỏng thì cỡ khác thường đúng), rồi mô hình nhẹ
    const s0=opts.size||256, sizes=[s0].concat([256,224,192].filter(z=>z!==s0));
    const order=(opts.fast||!sess.S)?[{k:fastK}]:sizes.map(z=>({k:'S',size:z})).concat([{k:fastK}]);
    let best=null;
    for(const o of order){
      const k=o.k; if(!sess[k]) continue;
      const q=await runModel(k,src,o.size); if(!valid(q,SW,SH)) continue;
      const tag=k+(o.size&&o.size!==256?o.size:''), sup=edgeSupport(q,P), mn=minOf(sup);
      info[tag]={sup,mn};
      // cạnh đối diện của một trang phẳng phải gần song song (chênh quá 25° là khung méo)
      const par=Math.max(angBetween([q[1][0]-q[0][0],q[1][1]-q[0][1]],[q[2][0]-q[3][0],q[2][1]-q[3][1]]),angBetween([q[3][0]-q[0][0],q[3][1]-q[0][1]],[q[2][0]-q[1][0],q[2][1]-q[1][1]]));
      const weak=[]; sup.forEach((v,i)=>{ if(v<0.5) weak.push(i); });
      if(par>25 && weak.length!==1){ info[tag].par=Math.round(par); if(!best) best={q,k:tag+'?',sup,mn:mn*0.5}; continue; }   // đúng 1 cạnh yếu thì để bước sửa cạnh gáy xử lý
      if(!weak.length){ best={q,k:tag,sup}; break; }
      if(weak.length===1){
        const e=weak[0], a=q[e], b=q[(e+1)%4], o1=q[(e+2)%4], o0=q[(e+3)%4];
        if(angBetween([b[0]-a[0],b[1]-a[1]],[o1[0]-o0[0],o1[1]-o0[1]])<10){ q.split=e; best={q,k:tag+'g',sup}; break; }   // cạnh gáy: yếu mép nhưng thẳng hàng
        const rq=repairEdge(q,sup,P);
        if(rq&&valid(rq,SW,SH)){ rq.conf=q.conf; best={q:rq,k:tag+'+sửa',sup:edgeSupport(rq,P)}; break; }
      }
      if(!best||mn>best.mn) best={q,k:tag+'?',sup,mn};
    }
    if(!best) return null;
    let q=best.q.map(p=>[Math.min(SW,Math.max(0,p[0])),Math.min(SH,Math.max(0,p[1]))]);
    if(best.q.split!==undefined){ q.split=best.q.split; }
    const cf=(best.q.conf||[]); q.minConf=cf.length?minOf(cf):1; q.meanConf=cf.length?cf.reduce((a,b)=>a+b,0)/cf.length:1;
    q.ai=best.k; q.minSup=minOf(best.sup); q.ms=Math.round(performance.now()-t0); q.info=info;
    return q;
  }catch(e){ console.warn('DocDet.core',e); return null; }
}
/* ---------- tìm trang bên cạnh / tách sổ 2 trang bằng chính mô hình AI ----------
   Mẹo: mô hình bắt rất tốt "trang giữa khung" khi trang bên kia bị khung ảnh cắt. Nên ta dựng 1 khung ảnh nắn thẳng, cắt sao cho
   trang cần tìm nằm giữa còn trang kia bị cắt, rồi hỏi mô hình. Không dựa vào nếp gấp (dễ nhầm với đường kẻ bảng). */
function solve8(A,b){
  const n=8;
  for(let i=0;i<n;i++){
    let m=i; for(let r=i+1;r<n;r++) if(Math.abs(A[r][i])>Math.abs(A[m][i])) m=r;
    [A[i],A[m]]=[A[m],A[i]]; [b[i],b[m]]=[b[m],b[i]];
    const d=A[i][i]||1e-12;
    for(let r=i+1;r<n;r++){ const f=A[r][i]/d; for(let c=i;c<n;c++) A[r][c]-=f*A[i][c]; b[r]-=f*b[i]; }
  }
  const x=new Array(n).fill(0);
  for(let i=n-1;i>=0;i--){ let s=b[i]; for(let c=i+1;c<n;c++) s-=A[i][c]*x[c]; x[i]=s/(A[i][i]||1e-12); }
  return x;
}
function homog(from,to){
  const A=[],b=[];
  for(let i=0;i<4;i++){ const [x,y]=from[i],[u,v]=to[i]; A.push([x,y,1,0,0,0,-x*u,-y*u]); b.push(u); A.push([0,0,0,x,y,1,-x*v,-y*v]); b.push(v); }
  return solve8(A,b);
}
const hmap=(h,x,y)=>{ const d=h[6]*x+h[7]*y+1; return [(h[0]*x+h[1]*y+h[2])/d,(h[3]*x+h[4]*y+h[5])/d]; };
let gC=null;
function grab(src,maxSide){
  const SW=src.videoWidth||src.width, SH=src.videoHeight||src.height, k=Math.min(1,maxSide/Math.max(SW,SH)), w=Math.round(SW*k), h=Math.round(SH*k);
  if(!gC) gC=document.createElement('canvas'); gC.width=w; gC.height=h;
  const x=gC.getContext('2d',{willReadFrequently:true}); x.imageSmoothingQuality='high'; x.drawImage(src,0,0,w,h);
  return {k,w,h,d:x.getImageData(0,0,w,h).data};
}
/* khung nắn thẳng: vùng [Xa,Xb]x[Ya,Yb] của hệ toạ độ "trang chữ nhật" (trang = [0,Wp]x[0,Hp]) */
function warpFrame(G,Hf,Xa,Xb,Ya,Yb,outMax){
  const sc=outMax/Math.max(Xb-Xa,Yb-Ya), W=Math.max(8,Math.round((Xb-Xa)*sc)), Hh=Math.max(8,Math.round((Yb-Ya)*sc));
  const cv=document.createElement('canvas'); cv.width=W; cv.height=Hh;
  const cx=cv.getContext('2d'), im=cx.createImageData(W,Hh), o=im.data, d=G.d, gw=G.w, gh=G.h;
  for(let v=0;v<Hh;v++) for(let u=0;u<W;u++){
    const [x0,y0]=hmap(Hf,Xa+(u+0.5)/sc,Ya+(v+0.5)/sc), x=x0*G.k-0.5, y=y0*G.k-0.5, oi=(v*W+u)*4;
    if(x<0||y<0||x>gw-1.001||y>gh-1.001){ o[oi]=50; o[oi+1]=45; o[oi+2]=45; o[oi+3]=255; continue; }
    const xi=x|0, yi=y|0, fx=x-xi, fy=y-yi, i0=(yi*gw+xi)*4, i1=i0+gw*4;
    for(let c=0;c<3;c++){ const t=d[i0+c]*(1-fx)+d[i0+4+c]*fx, bt=d[i1+c]*(1-fx)+d[i1+4+c]*fx; o[oi+c]=t*(1-fy)+bt*fy; }
    o[oi+3]=255;
  }
  cx.putImageData(im,0,0);
  return {cv,sc,Xa,Ya};
}
const quadDims=q=>({w:(dist(q[0],q[1])+dist(q[3],q[2]))/2, h:(dist(q[0],q[3])+dist(q[1],q[2]))/2});
const rectOf=(w,h)=>[[0,0],[w,0],[w,h],[0,h]];
/* chạy mô hình trên khung, trả 4 góc theo toạ độ "trang chữ nhật" */
async function runFrame(fr){
  const q=await core(fr.cv,{});
  if(!q) return null;
  return q.map(p=>[fr.Xa+p[0]/fr.sc, fr.Ya+p[1]/fr.sc]);
}
const inImg=(q,SW,SH,m)=>q.every(p=>p[0]>=m&&p[1]>=m&&p[0]<=SW-m&&p[1]<=SH-m);
/* tìm trang nằm bên ngoài cạnh e của trang P */
async function findNeighbor(G,P,e,SW,SH){
  const {w:Wp,h:Hp}=quadDims(P), Hf=homog(rectOf(Wp,Hp),P), horiz=(e===1||e===3), Ld=horiz?Wp:Hp, Lc=horiz?Hp:Wp;
  let Xa=-0.14*Wp, Xb=1.14*Wp, Ya=-0.14*Hp, Yb=1.14*Hp;
  if(e===1){ Xa=0.65*Wp; Xb=Wp+1.45*Wp; } else if(e===3){ Xa=-1.45*Wp; Xb=0.35*Wp; }
  else if(e===2){ Ya=0.65*Hp; Yb=Hp+1.45*Hp; } else { Ya=-1.45*Hp; Yb=0.35*Hp; }
  const fr=warpFrame(G,Hf,Xa,Xb,Ya,Yb,900), R=await runFrame(fr); if(!R) return null;
  // trong toạ độ trang: a = khoảng cách ra ngoài từ cạnh gáy, c = vị trí dọc cạnh gáy
  const A=R.map(p=>e===1?p[0]-Wp:e===3?-p[0]:e===2?p[1]-Hp:-p[1]), C=R.map(p=>horiz?p[1]:p[0]);
  const sa=A.slice().sort((x,y)=>x-y), near=(sa[0]+sa[1])/2, far=(sa[2]+sa[3])/2;
  if(DD.debug){ DD._dbg=DD._dbg||[]; DD._dbg.push({fn:'neighbor',e,near:+(near/Ld).toFixed(3),far:+((far-near)/Ld).toFixed(3),c:[+(Math.min.apply(null,C)/Lc).toFixed(2),+(Math.max.apply(null,C)/Lc).toFixed(2)],img:fr.cv.toDataURL('image/jpeg',0.6)}); }
  if(Math.abs(near)>0.10*Ld || far-near<0.5*Ld || far-near>1.7*Ld) return null;
  const c0=Math.min.apply(null,C), c1=Math.max.apply(null,C), ov=Math.min(c1,Lc)-Math.max(c0,0);
  if(ov<0.6*Lc || (c1-c0)>1.5*Lc) return null;
  const N=R.map(p=>hmap(Hf,p[0],p[1]));
  if(N.some(p=>!isFinite(p[0])||!isFinite(p[1]))) return null;
  N.split=(e+2)%4; N.rel=(far-near)/Ld; return N;
}
/* quad Q bao cả 2 trang: tách thành [trái, phải] bằng 2 khung (bên kia bị cắt) */
async function cutSpread(G,Q){
  const {w:Ws,h:Hs}=quadDims(Q), Hf=homog(rectOf(Ws,Hs),Q);
  // thử vài mức cắt: trang kia phải còn thấy đủ rộng để mô hình thấy nếp gáy, nhưng không bị coi là cùng 1 trang
  for(const u of [0.30,0.15]){
    const fr=warpFrame(G,Hf,u*Ws,1.14*Ws,-0.14*Hs,1.14*Hs,900), fl=warpFrame(G,Hf,-0.14*Ws,(1-u)*Ws,-0.14*Hs,1.14*Hs,900);
    const R=await runFrame(fr);
    if(!R||(R[0][0]+R[3][0])/2<=u*Ws+0.05*Ws){ if(DD.debug){ DD._dbg=DD._dbg||[]; DD._dbg.push({fn:'cut',u,R:R&&R.map(p=>p.map(Math.round)),skip:'R sát mép cắt'}); } continue; }   // khung bên phải không thấy gáy: bỏ qua, đỡ chạy thêm khung trái
    const Lq=await runFrame(fl);
    if(DD.debug){ DD._dbg=DD._dbg||[]; DD._dbg.push({fn:'cut',u,R:R&&R.map(p=>p.map(Math.round)),L:Lq&&Lq.map(p=>p.map(Math.round)),Ws:Math.round(Ws)}); }
    if(!R||!Lq) continue;
    const xr=(R[0][0]+R[3][0])/2, xl=(Lq[1][0]+Lq[2][0])/2;                       // cạnh gáy của mỗi trang
    const wr=(R[1][0]+R[2][0])/2-xr, wl=xl-(Lq[0][0]+Lq[3][0])/2;
    const okAdj=Math.abs(xr-xl)<=0.07*Ws, okW=wr>0.25*Ws&&wr<0.75*Ws&&wl>0.25*Ws&&wl<0.75*Ws;
    const okEnds=Math.abs((Lq[0][0]+Lq[3][0])/2)<=0.10*Ws && Math.abs((R[1][0]+R[2][0])/2-Ws)<=0.10*Ws;
    const notBorder=xr>u*Ws+0.05*Ws && xl<(1-u)*Ws-0.05*Ws;                      // cạnh gáy tìm được nằm trong khung, không phải mép cắt
    if(!(okAdj&&okW&&okEnds&&notBorder)) continue;
    const toSrc=q=>q.map(p=>hmap(Hf,p[0],p[1]));
    const pl=toSrc(Lq), pr=toSrc(R); pl.split=1; pr.split=3; pl.ratio=Math.min(wl,wr)/Math.max(wl,wr); return [pl,pr];
  }
  return null;
}
/* phương án dự phòng khi mô hình không tách được: tìm gáy bằng "bước nhảy độ sáng giữa 2 trang + nếp tối",
   chỉ nhận khi đủ mạnh và liên tục từ trên xuống dưới (đường kẻ bảng không có bước nhảy độ sáng nên không đạt) */
function creaseSplit(G,Q){
  const {w:Ws,h:Hs}=quadDims(Q), Hf=homog(rectOf(Ws,Hs),Q), fr=warpFrame(G,Hf,0,Ws,0,Hs,600), cv=fr.cv, W=cv.width, H=cv.height;
  const d=cv.getContext('2d').getImageData(0,0,W,H).data, strips=[];
  for(let s=0;s<5;s++){ const r0=Math.floor(H*(0.1+0.16*s)), r1=Math.floor(H*(0.1+0.16*(s+1))), p=new Float32Array(W);
    for(let x=0;x<W;x++){ let a=0; for(let y=r0;y<r1;y++){ const i=(y*W+x)*4; a+=0.299*d[i]+0.587*d[i+1]+0.114*d[i+2]; } p[x]=a/(r1-r0); }
    strips.push(p); }
  const w=Math.floor(W*0.06), g0=Math.max(1,Math.floor(W*0.012)), lo=Math.floor(W*0.25), hi=Math.floor(W*0.75), mean=(p,a,b)=>{ let s=0; for(let i=a;i<b;i++) s+=p[i]; return s/(b-a); };
  const res=strips.map(p=>{ const r=new Float32Array(W).fill(-1e9); for(let x=lo;x<hi;x++){ const L=mean(p,x-w,x-g0), R=mean(p,x+g0,x+w), c=mean(p,x-g0,x+g0+1); r[x]=Math.min(Math.abs(R-L),60)+2*Math.max(Math.min(L,R)-c,0); } return r; });
  let bx=-1,bv=-1;
  for(let x=lo+2;x<hi-2;x++){ let m=1e9; for(const r of res){ let s=0; for(let k=-2;k<=2;k++) s+=r[x+k]; m=Math.min(m,s/5); } if(m>bv){ bv=m; bx=x; } }
  const out={score:bv,pages:null};
  if(bv<7.5) return out;
  const f=bx/W, sh=0.005, xl=(f-sh)*Ws, xr=(f+sh)*Ws, toSrc=q=>q.map(p=>hmap(Hf,p[0],p[1]));
  const pl=toSrc([[0,0],[xl,0],[xl,Hs],[0,Hs]]), pr=toSrc([[xr,0],[Ws,0],[Ws,Hs],[xr,Hs]]);
  pl.split=1; pr.split=3; pl.ratio=Math.min(f,1-f)/Math.max(f,1-f); out.pages=[pl,pr]; return out;
}
/* cạnh e của quad có "giấy" ngay bên ngoài không (gợi ý có trang bên cạnh) */
function paperOutside(P,q,e){
  const a=q[e], b=q[(e+1)%4], o1=q[(e+2)%4], o0=q[(e+3)%4];
  const Le=dist(a,b); if(Le<10) return false;
  let nx=-(b[1]-a[1])/Le, ny=(b[0]-a[0])/Le;
  const cx=(q[0][0]+q[1][0]+q[2][0]+q[3][0])/4, cy=(q[0][1]+q[1][1]+q[2][1]+q[3][1])/4;
  if(((a[0]+b[0])/2-cx)*nx+((a[1]+b[1])/2-cy)*ny<0){ nx=-nx; ny=-ny; }
  const Ld=Math.abs(((a[0]+b[0])/2-(o0[0]+o1[0])/2)*nx+((a[1]+b[1])/2-(o0[1]+o1[1])/2)*ny);
  const at=(x,y)=>{ const xi=Math.round(x*P.s), yi=Math.round(y*P.s); if(xi<0||yi<0||xi>=P.w||yi>=P.h) return NaN; return P.g[yi*P.w+xi]; };
  const med=(a,mn)=>{ const z=a.filter(v=>!isNaN(v)).sort((p,r)=>p-r); return z.length>=mn?z[z.length>>1]:NaN; };
  const ins=[], outs=[];
  for(let i=0;i<5;i++){ const t=0.2+0.15*i, x0=a[0]+(b[0]-a[0])*t, y0=a[1]+(b[1]-a[1])*t;
    ins.push(at(x0-nx*Ld*0.25,y0-ny*Ld*0.25));
    for(const s of [0.10,0.25,0.40]) outs.push(at(x0+nx*Ld*s,y0+ny*Ld*s)); }
  const pi=med(ins,3), po=med(outs,8); if(DD.debug){ DD._dbg=DD._dbg||[]; DD._dbg.push({fn:'paperOutside',e,pi,po}); } return !isNaN(pi)&&!isNaN(po)&&po>=0.62*pi&&po>=110;
}
/* còn chỗ trong khung ảnh cho nguyên 1 trang bên ngoài cạnh e không (không thì bỏ qua cho đỡ tốn thời gian) */
function roomOutside(q,e,SW,SH){
  const a=q[e], b=q[(e+1)%4], o1=q[(e+2)%4], o0=q[(e+3)%4], Le=dist(a,b); if(Le<10) return 0;
  let nx=-(b[1]-a[1])/Le, ny=(b[0]-a[0])/Le; const cx=(q[0][0]+q[1][0]+q[2][0]+q[3][0])/4, cy=(q[0][1]+q[1][1]+q[2][1]+q[3][1])/4;
  const mx=(a[0]+b[0])/2, my=(a[1]+b[1])/2; if((mx-cx)*nx+(my-cy)*ny<0){ nx=-nx; ny=-ny; }
  const Ld=Math.abs((mx-(o0[0]+o1[0])/2)*nx+(my-(o0[1]+o1[1])/2)*ny); let t=0;
  while(t<3*Ld){ const x=mx+nx*t, y=my+ny*t; if(x<0||y<0||x>SW||y>SH) break; t+=Ld/20; }
  return t/Ld;
}
const inQuad=(q,x,y)=>{ let s=0; for(let i=0;i<4;i++){ const a=q[i],b=q[(i+1)%4], c=(b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]); if(c>0) s++; else if(c<0) s--; } return Math.abs(s)===4; };
const cen=q=>[(q[0][0]+q[1][0]+q[2][0]+q[3][0])/4,(q[0][1]+q[1][1]+q[2][1]+q[3][1])/4];
function pickPageIdx(pages,SW,SH){
  const c=[SW/2,SH/2], inside=pages.map(p=>inQuad(p,c[0],c[1]));
  if(inside[0]!==inside[1]) return inside[0]?0:1;
  const d=pages.map(p=>dist(cen(p),c)); return d[0]<=d[1]?0:1;
}
let spreadMemo=null;
async function trySpread(src,q,opts){
  const SW=src.videoWidth||src.width, SH=src.videoHeight||src.height, diag=Math.hypot(SW,SH);
  const now=performance.now();
  if(spreadMemo && now-spreadMemo.t<8000 && spreadMemo.none && Math.max.apply(null,q.map((p,i)=>dist(p,spreadMemo.q[i])))/diag<0.03) return null;
  const G=grab(src,1600), qd=quadDims(q); let pages=null, how='';
  if(qd.w/qd.h>=1.05){
    const cr=creaseSplit(G,q);                                   // chấm nhanh: quá phẳng (không có dấu hiệu gáy) thì khỏi chạy thêm mô hình
    if(cr.score>=4){ const cut=await cutSpread(G,q); if(cut){ pages=cut; how='cut'; } }
    if(!pages && qd.w/qd.h>=1.25 && cr.pages){ pages=cr.pages; how='crease'; }
  }
  if(!pages){
    const P=prep(src,opts.edgeSide||480), cand=[];
    for(let e=0;e<4;e++){ if((q.split===e || paperOutside(P,q,e)) && roomOutside(q,e,SW,SH)>=0.75) cand.push(e); }
    cand.sort((a,b)=>(a===q.split?-1:0)-(b===q.split?-1:0));
    for(const e of cand.slice(0,2)){
      const N=await findNeighbor(G,q,e,SW,SH);
      if(N){ const P2=q.map(p=>p.slice()); P2.split=e; pages=(e===1||e===2)?[P2,N]:[N,P2]; how='neighbor'; break; }
    }
  }
  if(!pages){ spreadMemo={t:now,q:q.map(p=>p.slice()),none:true}; return null; }
  spreadMemo=null;
  const m=-0.004*Math.max(SW,SH), nb=pages.find(p=>p.rel!==undefined), sizeOk=(how==='cut'||how==='crease')?pages[0].ratio>=0.6:(nb&&nb.rel>=0.6&&nb.rel<=1.4), full=pages.every(p=>inImg(p,SW,SH,m))&&!!sizeOk;
  return {pages,pick:pickPageIdx(pages,SW,SH),full,how};
}
DD._i.trySpread=trySpread; DD._i.core=core;
/* ---------- hàm chính ----------
   opts.fast: chỉ dùng mô hình nhẹ (xem trước khi mô hình mạnh chưa nạp). opts.spread=false: không tìm trang bên cạnh.
   Kết quả q = trang ở giữa; q.spread = {pages:[trái,phải], pick, full} khi thấy 2 trang. */
DD.detect = async function(src,opts){
  opts=opts||{};
  if(!DD.ready) return null;
  let q=await core(src,opts); if(!q) return null;
  let spread=null;
  if(sess.S && !opts.fast && opts.spread!==false){
    try{ spread=await trySpread(src,q,opts); }catch(e){ console.warn('DocDet.spread',e); }
  }
  const snapOne=pg=>{
    if(opts.snap===false) return pg;
    try{ const sq=snapEdges(src,pg,{side:opts.snapSide||(src.videoWidth?800:1400)}); sq.split=pg.split; return sq; }catch(e){ return pg; }
  };
  const keep=()=>({ai:q.ai,minConf:q.minConf,meanConf:q.meanConf,minSup:q.minSup,info:q.info,ms:q.ms});
  if(spread){
    const pages=spread.pages.map(pg=>{ const c=snapOne(pg); c.split=pg.split; return c; });
    const main=pages[spread.pick];
    Object.assign(main,keep()); main.pre=q;
    main.spread={pages,pick:spread.pick,full:spread.full,how:spread.how};
    return main;
  }
  if(!/\?/.test(q.ai)){ const sq=snapOne(q); Object.assign(sq,keep()); sq.pre=q; q=sq; }
  return q;
};

})();
