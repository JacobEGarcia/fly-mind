import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const bar = document.querySelector('#bar i');
const loadmsg = document.getElementById('loadmsg');
const setP = (p, m) => { bar.style.width = (p*100).toFixed(0)+'%'; if(m) loadmsg.textContent = m; };

async function fetchBin(url, as) {
  const r = await fetch(url);
  const b = await r.arrayBuffer();
  return as === 'f32' ? new Float32Array(b) : as === 'u32' ? new Uint32Array(b) :
         as === 'i8' ? new Int8Array(b) : new Uint8Array(b);
}

setP(.05, 'reading the wiring diagram…');
const meta = await (await fetch('meta.json')).json();
const N = meta.neuron_count, E = meta.edge_count;
const [positions, attrs, edgesU, wI8] = await Promise.all([
  fetchBin('positions.bin','f32'), fetchBin('attrs.bin','u8'),
  fetchBin('edges.bin','u32'), fetchBin('w.bin','i8')
]);
setP(.35, '139,255 neurons loaded — building the brain…');

// ---------- simulation state ----------
const act = new Float32Array(N);      // membrane-ish activity 0..~1.5
const next = new Float32Array(N);
const E_pre = new Uint32Array(E), E_post = new Uint32Array(E), E_w = new Float32Array(E);
for (let e = 0; e < E; e++) {
  E_pre[e] = edgesU[2*e]; E_post[e] = edgesU[2*e+1]; E_w[e] = wI8[e] / 140;
}
// adjacency (CSR) over pre neurons
const outStart = new Uint32Array(N+1);
for (let e = 0; e < E; e++) outStart[E_pre[e]+1]++;
for (let i = 0; i < N; i++) outStart[i+1] += outStart[i];
const csrPost = new Uint32Array(E), csrW = new Float32Array(E);
{
  const cursor = outStart.slice(0, N);
  for (let e = 0; e < E; e++) {
    const p = E_pre[e], o = cursor[p]++;
    csrPost[o] = E_post[e]; csrW[o] = E_w[e];
  }
}
setP(.5, 'mapping synapses…');

// ---------- three.js scene ----------
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x05060a, 0.0016);
const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.1, 4000);
camera.position.set(0, 90, 430);
const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.getElementById('scene').appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = .06;
controls.autoRotate = true; controls.autoRotateSpeed = .5;
controls.target.set(0, 30, 0);

const PALETTE = [ // super_class colors
  0x37e6ff, // optic - cyan
  0xffb300, // central - amber
  0xff4fd8, // sensory - magenta
  0x7dff5e, // visual_projection - green
  0xb98cff, // ascending - violet
  0xff3b30, // descending - red
  0xff9de2, // sensory_ascending
  0x8ff7e7, // visual_centrifugal
  0xffe14f, // motor - yellow
  0xffffff  // endocrine
];

// points
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const aClass = new Float32Array(N), aAct = new Float32Array(N);
for (let i = 0; i < N; i++) aClass[i] = attrs[3*i];
geo.setAttribute('aClass', new THREE.BufferAttribute(aClass, 1));
const actAttr = new THREE.BufferAttribute(aAct, 1); actAttr.setUsage(THREE.DynamicDrawUsage);
geo.setAttribute('aAct', actAttr);
const ptsMat = new THREE.ShaderMaterial({
  uniforms: { uPalette: { value: PALETTE.map(c => new THREE.Color(c)) }, uPx: { value: renderer.getPixelRatio() } },
  vertexShader: `
    attribute float aClass; attribute float aAct;
    uniform vec3 uPalette[10]; uniform float uPx;
    varying vec3 vC; varying float vA;
    void main(){
      vC = uPalette[int(aClass)]; vA = aAct;
      vec4 mv = modelViewMatrix * vec4(position,1.0);
      float s = (1.3 + aAct*4.5) * uPx * (300.0 / -mv.z);
      gl_PointSize = clamp(s, 1.0, 22.0);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    varying vec3 vC; varying float vA;
    void main(){
      vec2 d = gl_PointCoord - .5;
      float r = length(d);
      if (r > .5) discard;
      float glow = smoothstep(.5, .0, r);
      vec3 col = mix(vC*.55, vec3(1.0), vA*.85);
      gl_FragColor = vec4(col, glow * (.42 + vA*.58));
    }`,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
});
const brain = new THREE.Points(geo, ptsMat);
brain.position.y = 60;
scene.add(brain);

// edges subset (every 5th) as lines, brightness follows pre-neuron activity
const STRIDE = 5, RE = Math.floor(E / STRIDE);
const lpos = new Float32Array(RE*6), lact = new Float32Array(RE*2);
const edgePreIdx = new Uint32Array(RE);
for (let k = 0; k < RE; k++) {
  const e = k*STRIDE, p = E_pre[e], q = E_post[e];
  edgePreIdx[k] = p;
  lpos.set([positions[3*p],positions[3*p+1],positions[3*p+2],
            positions[3*q],positions[3*q+1],positions[3*q+2]], k*6);
}
const lgeo = new THREE.BufferGeometry();
lgeo.setAttribute('position', new THREE.BufferAttribute(lpos, 3));
const lactAttr = new THREE.BufferAttribute(lact, 1); lactAttr.setUsage(THREE.DynamicDrawUsage);
lgeo.setAttribute('aAct', lactAttr);
const lmat = new THREE.ShaderMaterial({
  vertexShader: `
    attribute float aAct; varying float vA;
    void main(){ vA = aAct; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: `
    varying float vA;
    void main(){ gl_FragColor = vec4(mix(vec3(.12,.3,.4), vec3(1.,.85,.4), vA), .05 + vA*.55); }`,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
});
const wires = new THREE.LineSegments(lgeo, lmat);
wires.position.y = 60;
scene.add(wires);
setP(.62, 'growing a virtual body…');

// ---------- the embodied fly ----------
const fly = new THREE.Group();
const flyMat = new THREE.MeshStandardMaterial({ color:0x1b2430, roughness:.35, metalness:.7 });
const body = new THREE.Mesh(new THREE.SphereGeometry(6, 24, 18), flyMat);
body.scale.set(1.7, 1, 1);
const head = new THREE.Mesh(new THREE.SphereGeometry(3.6, 20, 16), flyMat);
head.position.set(9.5, 1.2, 0);
const eyeMat = new THREE.MeshStandardMaterial({ color:0xff3b30, roughness:.15, metalness:.4, emissive:0x550000 });
const eyeL = new THREE.Mesh(new THREE.SphereGeometry(2.5, 16, 12), eyeMat);
eyeL.position.set(11, 2.2, 2.2);
const eyeR = eyeL.clone(); eyeR.position.z = -2.2;
const wingMat = new THREE.MeshStandardMaterial({ color:0x9fdcff, transparent:true, opacity:.28, side:THREE.DoubleSide });
const wingGeo = new THREE.CircleGeometry(6, 20); wingGeo.scale(1.8, .7, 1);
const wingL = new THREE.Mesh(wingGeo, wingMat);
wingL.position.set(1, 4.4, 3.4); wingL.rotation.set(-.5, 0, .3);
const wingR = new THREE.Mesh(wingGeo, wingMat);
wingR.position.set(1, 4.4, -3.4); wingR.rotation.set(.5, 0, .3);
fly.add(body, head, eyeL, eyeR, wingL, wingR);
// legs: 6 two-segment legs
const legs = [];
const legMat = new THREE.MeshStandardMaterial({ color:0x11161f, roughness:.5, metalness:.6 });
for (let s = -1; s <= 1; s += 2) for (let k = 0; k < 3; k++) {
  const hip = new THREE.Group();
  hip.position.set(5 - k*4.6, -2.4, s*4.2);
  const upper = new THREE.Mesh(new THREE.CylinderGeometry(.34,.28,5.4,6), legMat);
  upper.position.y = -2.7;
  const knee = new THREE.Group(); knee.position.y = -5.4;
  const lower = new THREE.Mesh(new THREE.CylinderGeometry(.24,.14,5.6,6), legMat);
  lower.position.y = -2.8; knee.add(lower);
  hip.add(upper, knee);
  hip.rotation.x = s * .85;
  fly.add(hip);
  legs.push({ hip, knee, phase: (k*2.1 + (s>0?0:1.05)), side: s });
}
// antennae
const antMat = legMat;
for (let s=-1;s<=1;s+=2){
  const a = new THREE.Mesh(new THREE.CylinderGeometry(.1,.06,4.5,5), antMat);
  a.position.set(12.4, 3.6, s*1.4); a.rotation.z = -.9; a.rotation.x = s*.3;
  fly.add(a);
}
fly.position.set(0, -68, 0); fly.scale.setScalar(1.35);
scene.add(fly);

// chamber
const chamber = new THREE.Group();
const glass = new THREE.Mesh(new THREE.BoxGeometry(90, 60, 90),
  new THREE.MeshPhysicalMaterial({ color:0x37e6ff, transparent:true, opacity:.05, roughness:.1, metalness:0, side:THREE.BackSide }));
const edgesBox = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(90,60,90)),
  new THREE.LineBasicMaterial({ color:0x37e6ff, transparent:true, opacity:.3 }));
const dish = new THREE.Mesh(new THREE.CylinderGeometry(46, 46, 1.6, 48),
  new THREE.MeshStandardMaterial({ color:0x0d1420, roughness:.4, metalness:.8 }));
dish.position.y = -31;
chamber.add(glass, edgesBox, dish);
chamber.position.set(0, -55, 0);
scene.add(chamber);

scene.add(new THREE.AmbientLight(0x334455, 1.4));
const key = new THREE.DirectionalLight(0xffd9a0, 2.2); key.position.set(120, 200, 90); scene.add(key);
const rim = new THREE.DirectionalLight(0x37e6ff, 1.4); rim.position.set(-140, 60, -120); scene.add(rim);
const brainLight = new THREE.PointLight(0xffb300, 60000, 500); brainLight.position.set(0, 60, 0); scene.add(brainLight);
setP(.78, 'wiring the connectome to the body…');

// ---------- stimulation ----------
const PRESETS = meta.presets;
let stim = null, stimName = 'none', stimT = 0;
function stimulate(name, idxArr, strength=1.0, dur=1400) {
  stim = { idxArr, strength, until: performance.now() + dur };
  stimName = name;
  document.getElementById('h-stim').textContent = name;
  document.querySelectorAll('.sbtn').forEach(b => b.classList.toggle('hot', b.dataset.p === name));
}
document.querySelectorAll('.sbtn').forEach(b => b.addEventListener('click', () => {
  const p = b.dataset.p;
  if (p === 'reset') { act.fill(0); stim = null; stimName='none';
    document.getElementById('h-stim').textContent='none';
    document.querySelectorAll('.sbtn').forEach(x=>x.classList.remove('hot')); return; }
  stimulate(p, PRESETS[p], p==='descend'||p==='motor' ? 1.6 : 1.0, 2200);
}));

// click-to-stimulate via ray proximity
const raycaster = new THREE.Raycaster();
const ptr = new THREE.Vector2();
let downAt = 0;
renderer.domElement.addEventListener('pointerdown', () => downAt = performance.now());
renderer.domElement.addEventListener('pointerup', (ev) => {
  if (performance.now() - downAt > 220) return; // was a drag
  ptr.set((ev.clientX/innerWidth)*2-1, -(ev.clientY/innerHeight)*2+1);
  raycaster.setFromCamera(ptr, camera);
  const ro = raycaster.ray.origin, rd = raycaster.ray.direction;
  // nearest neuron to the ray (brain offset y=60)
  let best = -1, bestD = 12; // pick radius
  for (let i = 0; i < N; i++) {
    const dx = positions[3*i]-ro.x, dy = positions[3*i+1]+60-ro.y, dz = positions[3*i+2]-ro.z;
    const t = dx*rd.x + dy*rd.y + dz*rd.z;
    if (t < 0) continue;
    const px = dx - t*rd.x, py = dy - t*rd.y, pz = dz - t*rd.z;
    const d = Math.sqrt(px*px+py*py+pz*pz);
    if (d < bestD) { bestD = d; best = i; }
  }
  if (best >= 0) {
    // stimulate neighborhood: all neurons within 14 units of the pick
    const bx = positions[3*best], by = positions[3*best+1], bz = positions[3*best+2];
    const nb = [];
    for (let i = 0; i < N; i += 3) { // stride sample for speed, still ~46k checks
      const dx = positions[3*i]-bx, dy = positions[3*i+1]-by, dz = positions[3*i+2]-bz;
      if (dx*dx+dy*dy+dz*dz < 196) nb.push(i);
    }
    stimulate('local probe', nb, 1.4, 1600);
    toast('PROBE → ' + nb.length + ' NEIGHBORS');
  }
});
let toastT;
function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.style.opacity = 1;
  clearTimeout(toastT); toastT = setTimeout(() => t.style.opacity = 0, 1400);
}
document.querySelector('#info .tog').addEventListener('click',
  () => document.getElementById('info').classList.toggle('open'));

// ---------- motor readout (drives the fly) ----------
const motorSet = new Uint32Array([...PRESETS.descend, ...PRESETS.motor, ...PRESETS.ascending]);
function motorDrive() {
  let s = 0;
  for (let i = 0; i < motorSet.length; i++) s += act[motorSet[i]];
  return Math.min(s / (motorSet.length * 0.16), 1.6);
}
let driveSm = 0, walkPhase = 0, flyHeading = 0;

// ---------- main loop ----------
const clock = new THREE.Clock();
let spikes = 0, spikeCountWin = 0, lastHud = 0, activeCount = 0;
setP(1, 'brain online.');
document.getElementById('loading').style.display = 'none';

function step(dt) {
  // decay + stimulus injection
  const decay = Math.pow(0.06, dt); // fast-ish
  for (let i = 0; i < N; i++) act[i] *= decay;
  const now = performance.now();
  if (stim && now < stim.until) {
    const arr = stim.idxArr, s = stim.strength * dt * 9;
    for (let k = 0; k < arr.length; k++) act[arr[k]] = Math.min(act[arr[k]] + s * (0.6 + Math.random()*0.8), 1.4);
  } else if (stim) { stim = null; stimName='none';
    document.getElementById('h-stim').textContent='none';
    document.querySelectorAll('.sbtn').forEach(x=>x.classList.remove('hot'));
  }
  // propagate along real wiring
  spikes = 0; activeCount = 0;
  next.fill(0);
  for (let i = 0; i < N; i++) {
    const a = act[i];
    if (a > 0.35) {
      spikes++;
      const s = outStart[i], e2 = outStart[i+1];
      for (let o = s; o < e2; o++) next[csrPost[o]] += a * csrW[o];
    }
  }
  for (let i = 0; i < N; i++) {
    let v = act[i] + next[i];
    if (v > 1.4) v = 1.4; else if (v < 0.02) v = 0;
    act[i] = v;
    if (v > 0.12) activeCount++;
  }
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  step(dt);

  // push activity to GPU
  for (let i = 0; i < N; i++) aAct[i] = act[i];
  actAttr.needsUpdate = true;
  for (let k = 0; k < RE; k++) {
    const a = act[edgePreIdx[k]];
    lact[2*k] = a; lact[2*k+1] = a*0.6;
  }
  lactAttr.needsUpdate = true;

  // fly behavior
  const drive = motorDrive();
  driveSm += (drive - driveSm) * 0.06;
  const d = driveSm;
  const bEl = document.getElementById('behavior');
  let behavior = 'IDLE';
  if (d > 0.55) behavior = 'FLYING'; else if (d > 0.2) behavior = 'WALKING'; else if (d > 0.06) behavior = 'TWITCHING';
  bEl.textContent = behavior;
  walkPhase += dt * (2 + d * 10);
  const amp = Math.min(d, 1) * 0.9;
  for (const L of legs) {
    const ph = walkPhase + L.phase;
    L.hip.rotation.y = Math.sin(ph) * .55 * amp;
    L.knee.rotation.y = Math.max(0, Math.sin(ph + .9)) * .8 * amp;
  }
  if (behavior === 'WALKING' || behavior === 'FLYING') {
    flyHeading += dt * .45 * d;
    fly.position.x = Math.cos(flyHeading) * 24;
    fly.position.z = Math.sin(flyHeading) * 24;
    fly.rotation.y = -flyHeading;
  }
  const buzz = behavior === 'FLYING' ? 1 : 0;
  const t = performance.now() * .001;
  wingL.rotation.y = buzz ? Math.sin(t*120)*.5 : Math.sin(t*2)*.08;
  wingR.rotation.y = buzz ? -Math.sin(t*120)*.5 : -Math.sin(t*2)*.08;
  fly.position.y = -68 + (buzz ? 22 + Math.sin(t*7)*2 : Math.sin(t*1.3)*.5);
  brainLight.intensity = 40000 + Math.min(activeCount/N*8,1) * 120000;

  // HUD
  if (performance.now() - lastHud > 400) {
    lastHud = performance.now();
    document.getElementById('h-active').textContent = activeCount.toLocaleString();
    document.getElementById('h-spikes').textContent = Math.round(spikes/Math.max(dt,1e-3)*0 + spikes*2.5).toLocaleString();
  }
  controls.update();
  renderer.render(scene, camera);
}
animate();

addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// auto demo: after 2s, flash the optic lobes once
setTimeout(() => stimulate('optic', PRESETS.optic, 1.0, 1800), 2200);
