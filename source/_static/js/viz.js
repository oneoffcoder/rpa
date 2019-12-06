$(document).ready(function() {
  function getConsonants() {
      let C = 'c, d, f, h, k, l, m, n, p, q, r, s, t, v, x, y, z,';
      C += 'ch, dh, hl, hm, hn, kh, ml, nc, nk, np, nq, nr, nt, ny, ph, pl, qh, rh, th, ts, tx, xy,';
      C += 'hml, hny, nch, nkh, nph, npl, nqh, nrh, nth, nts, ntx, plh, tsh, txh,';
      C += 'nplh, ntsh, ntxh';
      C = C.split(',').map(c => c.trim());
    
      return C;
    }
    
    function getVowels() {
      let V = 'a, e, i, o, u, w, ai, au, aw, ee, ia, oo, ua'.split(',')
      .map(v => v.trim());
      return V;
    }
    
    function getTones() {
      let T = 'j, s, v, m, g, b, *'.split(',')
      .map(t => t.trim());
      return T;
    }
    
    function getSyllables() {
      const C = getConsonants();
      const V = getVowels();
      const T = getTones();
      const S = [];
    
      C.forEach(c => {
          V.forEach(v => {
            const I = [];
          T.forEach(t => {
            const cvt = `${c}${v}${t}`.replace('*', '');
            I.push(cvt)
          });
          S.push(I);
        });    
      });
    
      return S;
    }
    
    function getLargestWidth(font) {
        const S = getSyllables();
      let width = 0.0;
      S.forEach(I => {
          I.forEach(message => {
            const shape = font.generateShapes(message, 1);
          const geom = new THREE.ShapeBufferGeometry(shape);
          geom.computeBoundingBox();
    
          const w = (geom.boundingBox.max.x - geom.boundingBox.min.x);
          if (w > width) {
              width = w;
          }
        });
      });
      return width;
    }
    
    function getSMeshes(font, mat, delta) {
        const C = getConsonants();
      const V = getVowels();
      const T = getTones();
      const M = [];
      
      let y = 0.0;
      C.forEach(c => {
          let x = 0.0;
          V.forEach(v => {
            let z = 0.0;
            T.forEach(t => {
            const message = `${c}${v}${t}`.replace('*', '');
            
            const shape = font.generateShapes(message, 1);
            const geom = new THREE.ShapeBufferGeometry(shape);
            geom.computeBoundingBox();
            geom.translate(x, y, z);
            const mesh = new THREE.Mesh(geom, mat);
            M.push(mesh);
    
            z += 1.2;
          });
          x += delta;
        });    
        y += 1.2;
      });
      
      return M;
    }
    
    function getCMeshes(font, mat) {
        const C = getConsonants();
      const M = []
      const yDelta = 1.2;
      let yPos = 0;
      C.forEach(message => {
          const shape = font.generateShapes(message, 1);
        const geom = new THREE.ShapeBufferGeometry(shape);
        geom.computeBoundingBox();
        
        const xPos = -(geom.boundingBox.max.x - geom.boundingBox.min.x);
        geom.translate(xPos, yPos, 0);
        
        const mesh = new THREE.Mesh(geom, mat);
        M.push(mesh);
        
        yPos += yDelta;
      });
      return M;
    }
    
    function getVMeshes(font, mat, yDelta) {
        const V = getVowels();
      const M = [];
      let yPos = 0;
      V.forEach(message => {
          const shape = font.generateShapes(message, 1);
        const geom = new THREE.ShapeBufferGeometry(shape);
        geom.computeBoundingBox();
        
        const xPos = -(geom.boundingBox.max.x - geom.boundingBox.min.x);
        geom.translate(xPos, yPos, 0);
        
        const mesh = new THREE.Mesh(geom, mat);
        mesh.rotation.x = -1.5708;
          mesh.rotation.z = -1.5708;
        M.push(mesh);
        
        yPos += yDelta;
      });
      return M;
    }
    
    function getTMeshes(font, mat, yDelta) {
        const T = getTones();
      const M = [];
      let yPos = 0;
      T.forEach(message => {
          const shape = font.generateShapes(message, 1);
        const geom = new THREE.ShapeBufferGeometry(shape);
        geom.computeBoundingBox();
        
        const xPos = (geom.boundingBox.max.x - geom.boundingBox.min.x);
        geom.translate(0.5, yPos, 0);
        
        const mesh = new THREE.Mesh(geom, mat);
        mesh.rotation.x = -1.5708;
          mesh.rotation.z = -3.1415;
        M.push(mesh);
        
        yPos += yDelta;
      });
      return M;
    }

    function start(callback) {
      const container = document.getElementById('viz-canvas');
      
      var renderer = new THREE.WebGLRenderer();
      renderer.setSize( 700, 700 );
      container.appendChild( renderer.domElement );
      
      var camera = new THREE.PerspectiveCamera( 20, window.innerWidth / window.innerHeight, 1, 500 );
      camera.position.set( 20, 10, 20 );
      //camera.position.set( 0, 0, 20 );
      camera.lookAt( 0, 0, 0 );
      
      var controls = new THREE.OrbitControls( camera, renderer.domElement );
      controls.update();
      
      var scene = new THREE.Scene();
      
      var axesHelper = new THREE.AxesHelper( 5 );
      scene.add( axesHelper );
      
      var loader = new THREE.FontLoader();
      loader.load( 'https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', function ( font ) {
          var color = new THREE.Color( 0x006699 );
          var matLite = new THREE.MeshBasicMaterial( {
          color: color,
          transparent: true,
          opacity: 0.4,
          side: THREE.DoubleSide
        } );
          
        const width = 5.583000183105469; //getLargestWidth(font);
        //console.log(width);
        
        const cMeshes = getCMeshes(font, matLite);
        cMeshes.forEach(m => scene.add(m));
        
        const vMeshes = getVMeshes(font, matLite, width);
        vMeshes.forEach(m => scene.add(m));
        
        const tMeshes = getTMeshes(font, matLite, 1.2);
        tMeshes.forEach(m => scene.add(m));
        
        const sMeshes = getSMeshes(font, matLite, width);
        sMeshes.forEach(m => scene.add(m));
        
        console.log('done');
        if (callback) {
          callback();
        }
        
      });
      
      function animate() {
      
          requestAnimationFrame( animate );
      
          // required if controls.enableDamping or controls.autoRotate are set to true
          controls.update();
      
          renderer.render( scene, camera );
      
      }
      
      animate();
    }

    const canvas = $('#viz-canvas');
    if (canvas.length > 0) {
      const btn = $('#viz-btn');
      const msg = $('#viz-msg');
      
      btn.click(() => {
        console.log('starting viz now');
        msg.css('visibility', 'visible');

        start(() => {
          console.log('callback');
          btn.toggle();
          msg.toggle();
        });
      });
    }
    
    
    
});