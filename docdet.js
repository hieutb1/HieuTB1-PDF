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
DD.detect = async function(src,opts){
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
    if(opts.snap!==false && !/\?/.test(best.k)){
      try{ const sq=snapEdges(src,q,{side:opts.snapSide||(src.videoWidth?800:1400)}); sq.split=q.split; sq.minConf=q.minConf; sq.meanConf=q.meanConf; sq.pre=q; q=sq; }catch(e){ console.warn('DocDet.snap',e); }
    }
    q.ai=best.k; q.minSup=minOf(best.sup); q.ms=Math.round(performance.now()-t0); q.info=info;
    return q;
  }catch(e){ console.warn('DocDet.detect',e); return null; }
};
})();
