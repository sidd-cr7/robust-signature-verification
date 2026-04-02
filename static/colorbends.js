const MAX_COLORS = 8;

const fragShader = `
#define MAX_COLORS ${MAX_COLORS}
uniform vec2 uCanvas;
uniform float uTime;
uniform float uSpeed;
uniform vec2 uRot;
uniform int uColorCount;
uniform vec3 uColors[MAX_COLORS];
uniform int uTransparent;
uniform float uScale;
uniform float uFrequency;
uniform float uWarpStrength;
uniform vec2 uPointer;
uniform float uMouseInfluence;
uniform float uParallax;
uniform float uNoise;
varying vec2 vUv;

void main() {
  float t = uTime * uSpeed;
  vec2 p = vUv * 2.0 - 1.0;
  p += uPointer * uParallax * 0.1;
  vec2 rp = vec2(p.x*uRot.x - p.y*uRot.y, p.x*uRot.y + p.y*uRot.x);
  vec2 q = vec2(rp.x*(uCanvas.x/uCanvas.y), rp.y);
  q /= max(uScale, 0.0001);
  q /= 0.5 + 0.2*dot(q,q);
  q += 0.2*cos(t) - 7.56;
  vec2 toward = (uPointer - rp);
  q += toward * uMouseInfluence * 0.2;

  vec3 col = vec3(0.0);
  float a = 1.0;

  vec2 s = q;
  vec3 sumCol = vec3(0.0);
  float cover = 0.0;
  for (int i = 0; i < MAX_COLORS; ++i) {
    if (i >= uColorCount) break;
    s -= 0.01;
    vec2 r = sin(1.5*(s.yx*uFrequency) + 2.0*cos(s*uFrequency));
    float m0 = length(r + sin(5.0*r.y*uFrequency - 3.0*t + float(i))/4.0);
    float kBelow = clamp(uWarpStrength, 0.0, 1.0);
    float kMix = pow(kBelow, 0.3);
    float gain = 1.0 + max(uWarpStrength - 1.0, 0.0);
    vec2 disp = (r - s)*kBelow;
    vec2 warped = s + disp*gain;
    float m1 = length(warped + sin(5.0*warped.y*uFrequency - 3.0*t + float(i))/4.0);
    float m = mix(m0, m1, kMix);
    float w = 1.0 - exp(-6.0/exp(6.0*m));
    sumCol += uColors[i]*w;
    cover = max(cover, w);
  }
  col = clamp(sumCol, 0.0, 1.0);
  a = cover;

  if (uNoise > 0.0001) {
    float n = fract(sin(dot(gl_FragCoord.xy + vec2(uTime), vec2(12.9898,78.233)))*43758.5453123);
    col += (n - 0.5)*uNoise;
    col = clamp(col, 0.0, 1.0);
  }

  gl_FragColor = vec4(col * a, a);
}`;

const vertShader = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 1.0);
}`;

function initColorBends(canvas, opts = {}) {
    const cfg = {
        colors:        opts.colors        ?? ['#2563eb','#7c3aed','#1e1b4b','#0f0f1a','#4f46e5','#6d28d9'],
        speed:         opts.speed         ?? 0.2,
        rotation:      opts.rotation      ?? 45,
        scale:         opts.scale         ?? 1.0,
        frequency:     opts.frequency     ?? 1.0,
        warpStrength:  opts.warpStrength  ?? 1.0,
        mouseInfluence:opts.mouseInfluence?? 0.8,
        parallax:      opts.parallax      ?? 0.5,
        noise:         opts.noise         ?? 0.05,
    };

    const THREE = window.THREE;
    if (!THREE) { console.error('Three.js not loaded'); return; }

    const scene    = new THREE.Scene();
    const camera   = new THREE.OrthographicCamera(-1,1,1,-1,0,1);
    const geometry = new THREE.PlaneGeometry(2,2);

    function hexToVec3(hex) {
        const h = hex.replace('#','');
        return new THREE.Vector3(
            parseInt(h.slice(0,2),16)/255,
            parseInt(h.slice(2,4),16)/255,
            parseInt(h.slice(4,6),16)/255
        );
    }

    const uColorsArr = Array.from({length: MAX_COLORS}, () => new THREE.Vector3());
    const colorVecs  = cfg.colors.slice(0, MAX_COLORS).map(hexToVec3);
    colorVecs.forEach((v,i) => uColorsArr[i].copy(v));

    const rad = cfg.rotation * Math.PI / 180;

    const material = new THREE.ShaderMaterial({
        vertexShader:   vertShader,
        fragmentShader: fragShader,
        transparent:    true,
        premultipliedAlpha: true,
        uniforms: {
            uCanvas:        { value: new THREE.Vector2(1,1) },
            uTime:          { value: 0 },
            uSpeed:         { value: cfg.speed },
            uRot:           { value: new THREE.Vector2(Math.cos(rad), Math.sin(rad)) },
            uColorCount:    { value: colorVecs.length },
            uColors:        { value: uColorsArr },
            uTransparent:   { value: 1 },
            uScale:         { value: cfg.scale },
            uFrequency:     { value: cfg.frequency },
            uWarpStrength:  { value: cfg.warpStrength },
            uPointer:       { value: new THREE.Vector2(0,0) },
            uMouseInfluence:{ value: cfg.mouseInfluence },
            uParallax:      { value: cfg.parallax },
            uNoise:         { value: cfg.noise },
        }
    });

    scene.add(new THREE.Mesh(geometry, material));

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    function resize() {
        const w = window.innerWidth, h = window.innerHeight;
        renderer.setSize(w, h, false);
        material.uniforms.uCanvas.value.set(w, h);
    }
    window.addEventListener('resize', resize);
    resize();

    const pointerTarget  = new THREE.Vector2(0,0);
    const pointerCurrent = new THREE.Vector2(0,0);
    window.addEventListener('mousemove', e => {
        pointerTarget.set(
            (e.clientX / window.innerWidth)  * 2 - 1,
           -((e.clientY / window.innerHeight) * 2 - 1)
        );
    });

    let start = null, raf;
    function frame(ts) {
        raf = requestAnimationFrame(frame);
        if (!start) start = ts;
        const elapsed = (ts - start) * 0.001;
        material.uniforms.uTime.value = elapsed;
        pointerCurrent.lerp(pointerTarget, 0.05);
        material.uniforms.uPointer.value.copy(pointerCurrent);
        renderer.render(scene, camera);
    }
    raf = requestAnimationFrame(frame);

    return () => {
        cancelAnimationFrame(raf);
        window.removeEventListener('resize', resize);
        geometry.dispose();
        material.dispose();
        renderer.dispose();
    };
}

window.initColorBends = initColorBends;
