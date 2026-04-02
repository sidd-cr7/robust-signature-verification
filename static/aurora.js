const vertexShader = `
attribute vec2 position;
attribute vec2 uv;
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 0, 1);
}`;

const fragmentShader = `
precision highp float;
uniform float uTime;
uniform vec3 uResolution;
uniform float uSpeed;
uniform float uScale;
uniform float uBrightness;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform float uNoiseFreq;
uniform float uNoiseAmp;
uniform float uBandHeight;
uniform float uBandSpread;
uniform float uOctaveDecay;
uniform float uLayerOffset;
uniform float uColorSpeed;
uniform vec2 uMouse;
uniform float uMouseInfluence;

#define TAU 6.28318

vec3 gradientHash(vec3 p) {
  p = vec3(dot(p,vec3(127.1,311.7,234.6)),dot(p,vec3(269.5,183.3,198.3)),dot(p,vec3(169.5,283.3,156.9)));
  vec3 h = fract(sin(p)*43758.5453123);
  float phi = acos(2.0*h.x-1.0);
  float theta = TAU*h.y;
  return vec3(cos(theta)*sin(phi),sin(theta)*cos(phi),cos(phi));
}

float quinticSmooth(float t) {
  float t2=t*t; float t3=t*t2;
  return 6.0*t3*t2-15.0*t2*t2+10.0*t3;
}

vec3 cosineGradient(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
  return a+b*cos(TAU*(c*t+d));
}

float perlin3D(float amplitude,float frequency,float px,float py,float pz) {
  float x=px*frequency; float y=py*frequency;
  float fx=floor(x); float fy=floor(y); float fz=floor(pz);
  float cx=ceil(x);  float cy=ceil(y);  float cz=ceil(pz);
  vec3 g000=gradientHash(vec3(fx,fy,fz)); vec3 g100=gradientHash(vec3(cx,fy,fz));
  vec3 g010=gradientHash(vec3(fx,cy,fz)); vec3 g110=gradientHash(vec3(cx,cy,fz));
  vec3 g001=gradientHash(vec3(fx,fy,cz)); vec3 g101=gradientHash(vec3(cx,fy,cz));
  vec3 g011=gradientHash(vec3(fx,cy,cz)); vec3 g111=gradientHash(vec3(cx,cy,cz));
  float d000=dot(g000,vec3(x-fx,y-fy,pz-fz)); float d100=dot(g100,vec3(x-cx,y-fy,pz-fz));
  float d010=dot(g010,vec3(x-fx,y-cy,pz-fz)); float d110=dot(g110,vec3(x-cx,y-cy,pz-fz));
  float d001=dot(g001,vec3(x-fx,y-fy,pz-cz)); float d101=dot(g101,vec3(x-cx,y-fy,pz-cz));
  float d011=dot(g011,vec3(x-fx,y-cy,pz-cz)); float d111=dot(g111,vec3(x-cx,y-cy,pz-cz));
  float sx=quinticSmooth(x-fx); float sy=quinticSmooth(y-fy); float sz=quinticSmooth(pz-fz);
  float lx00=mix(d000,d100,sx); float lx10=mix(d010,d110,sx);
  float lx01=mix(d001,d101,sx); float lx11=mix(d011,d111,sx);
  float ly0=mix(lx00,lx10,sy);  float ly1=mix(lx01,lx11,sy);
  return amplitude*mix(ly0,ly1,sz);
}

float auroraGlow(float t, vec2 shift) {
  vec2 uv = gl_FragCoord.xy / uResolution.y;
  uv += shift;
  float noiseVal = 0.0;
  float freq = uNoiseFreq;
  float amp = uNoiseAmp;
  vec2 sp = uv * uScale;
  for (float i = 0.0; i < 3.0; i += 1.0) {
    noiseVal += perlin3D(amp, freq, sp.x, sp.y, t);
    amp *= uOctaveDecay;
    freq *= 2.0;
  }
  float yBand = uv.y * 10.0 - uBandHeight * 10.0;
  return 0.3 * max(exp(uBandSpread * (1.0 - 1.1 * abs(noiseVal + yBand))), 0.0);
}

void main() {
  vec2 uv = gl_FragCoord.xy / uResolution.xy;
  float t = uSpeed * 0.4 * uTime;
  vec2 shift = (uMouse - 0.5) * uMouseInfluence;
  vec3 col = vec3(0.0);
  col += 0.99 * auroraGlow(t, shift) * cosineGradient(uv.x + uTime*uSpeed*0.2*uColorSpeed, vec3(0.5),vec3(0.5),vec3(1.0),vec3(0.3,0.20,0.20)) * uColor1;
  col += 0.99 * auroraGlow(t + uLayerOffset, shift) * cosineGradient(uv.x + uTime*uSpeed*0.1*uColorSpeed, vec3(0.5),vec3(0.5),vec3(2.0,1.0,0.0),vec3(0.5,0.20,0.25)) * uColor2;
  col *= uBrightness;
  float alpha = clamp(length(col), 0.0, 1.0);
  gl_FragColor = vec4(col, alpha);
}`;

function hexToVec3(hex) {
    const h = hex.replace('#', '');
    return [parseInt(h.slice(0,2),16)/255, parseInt(h.slice(2,4),16)/255, parseInt(h.slice(4,6),16)/255];
}

function initAurora(canvas, opts = {}) {
    const cfg = {
        speed: opts.speed ?? 0.6,
        scale: opts.scale ?? 1.5,
        brightness: opts.brightness ?? 1.2,
        color1: opts.color1 ?? '#4f46e5',
        color2: opts.color2 ?? '#7c3aed',
        noiseFreq: opts.noiseFreq ?? 2.5,
        noiseAmp: opts.noiseAmp ?? 1.0,
        bandHeight: opts.bandHeight ?? 0.5,
        bandSpread: opts.bandSpread ?? 1.0,
        octaveDecay: opts.octaveDecay ?? 0.1,
        layerOffset: opts.layerOffset ?? 0,
        colorSpeed: opts.colorSpeed ?? 1.0,
        mouseInfluence: opts.mouseInfluence ?? 0.25,
    };

    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) return;

    function compile(type, src) {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        return s;
    }

    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vertexShader));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fragmentShader));
    gl.linkProgram(prog);
    gl.useProgram(prog);

    // Full-screen triangle
    const verts = new Float32Array([-1,-1, 3,-1, -1,3]);
    const uvs   = new Float32Array([0,0, 2,0, 0,2]);

    const vBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vBuf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    const aPos = gl.getAttribLocation(prog, 'position');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    const uvBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
    gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);
    const aUv = gl.getAttribLocation(prog, 'uv');
    gl.enableVertexAttribArray(aUv);
    gl.vertexAttribPointer(aUv, 2, gl.FLOAT, false, 0, 0);

    const u = name => gl.getUniformLocation(prog, name);

    function resize() {
        canvas.width  = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.uniform3f(u('uResolution'), canvas.width, canvas.height, canvas.width / canvas.height);
    }
    window.addEventListener('resize', resize);
    resize();

    gl.uniform1f(u('uSpeed'),          cfg.speed);
    gl.uniform1f(u('uScale'),          cfg.scale);
    gl.uniform1f(u('uBrightness'),     cfg.brightness);
    gl.uniform3fv(u('uColor1'),        hexToVec3(cfg.color1));
    gl.uniform3fv(u('uColor2'),        hexToVec3(cfg.color2));
    gl.uniform1f(u('uNoiseFreq'),      cfg.noiseFreq);
    gl.uniform1f(u('uNoiseAmp'),       cfg.noiseAmp);
    gl.uniform1f(u('uBandHeight'),     cfg.bandHeight);
    gl.uniform1f(u('uBandSpread'),     cfg.bandSpread);
    gl.uniform1f(u('uOctaveDecay'),    cfg.octaveDecay);
    gl.uniform1f(u('uLayerOffset'),    cfg.layerOffset);
    gl.uniform1f(u('uColorSpeed'),     cfg.colorSpeed);
    gl.uniform1f(u('uMouseInfluence'), cfg.mouseInfluence);

    let mouse = [0.5, 0.5], target = [0.5, 0.5];
    window.addEventListener('mousemove', e => {
        target = [e.clientX / window.innerWidth, 1 - e.clientY / window.innerHeight];
    });

    let raf;
    function frame(t) {
        raf = requestAnimationFrame(frame);
        mouse[0] += 0.05 * (target[0] - mouse[0]);
        mouse[1] += 0.05 * (target[1] - mouse[1]);
        gl.uniform1f(u('uTime'), t * 0.001);
        gl.uniform2fv(u('uMouse'), mouse);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    raf = requestAnimationFrame(frame);

    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); };
}

window.initAurora = initAurora;
