(() => {
    'use strict';

    const explorer = document.getElementById('rpa-explorer');
    if (!explorer) return;

    const consonants = [
        '',
        ...('c d f h k l m n p q r s t v x y z ' +
            'ch dh hl hm hn kh ml nc nk np nq nr nt ny ph pl qh rh th ts tx xy ' +
            'hml hny nch nkh nph npl nqh nrh nth nts ntx plh tsh txh nplh ntsh ntxh').split(' ')
    ];
    const vowels = 'a e i o u w ai au aw ee ia oo ua'.split(' ');
    const tones = [
        { marker: 'j', name: 'High falling' },
        { marker: 's', name: 'Low' },
        { marker: 'v', name: 'Mid rising' },
        { marker: 'm', name: 'Creaky' },
        { marker: 'g', name: 'Low falling breathy' },
        { marker: 'b', name: 'High' },
        { marker: '', name: 'Mid' }
    ];
    const syllables = consonants.flatMap((consonant, c) =>
        vowels.flatMap((vowel, v) =>
            tones.map(({ marker }, t) => ({
                c, v, t, spelling: consonant + vowel + marker
            }))
        )
    );
    const bySpelling = new Map(syllables.map(item => [item.spelling, item]));

    const get = id => explorer.querySelector('#rpa-' + id);
    const search = get('search');
    const searchStatus = get('search-status');
    const consonantSelect = get('consonant');
    const vowelSelect = get('vowel');
    const toneSelect = get('tone');
    const viewSelect = get('view');
    const breakdown = get('breakdown');
    const playAll = get('play-all');
    const playbackStatus = get('playback');
    const viewSummary = get('view-summary');
    const stage = get('stage');
    const sceneStatus = get('scene-status');
    const resetView = get('reset-view');
    const count = new Intl.NumberFormat('en');

    let selection = bySpelling.get('po');
    let frame;
    let chart;
    let playback;

    const consonantLabel = c => consonants[c] || 'No consonant';
    const toneLabel = t => {
        const { marker, name } = tones[t];
        return marker ? marker + ' — ' + name : name + ' (no marker)';
    };
    const toneGroup = item => {
        const offset = (item.c * vowels.length + item.v) * tones.length;
        return syllables.slice(offset, offset + tones.length);
    };

    consonants.forEach((_, c) => consonantSelect.add(new Option(consonantLabel(c))));
    vowels.forEach(vowel => vowelSelect.add(new Option(vowel)));
    tones.forEach((_, t) => toneSelect.add(new Option(toneLabel(t))));

    const recordings = get('recordings');
    const toneButtons = tones.map(({ name }, t) => {
        const button = document.createElement('button');
        button.type = 'button';
        const spelling = document.createElement('strong');
        const caption = document.createElement('small');
        caption.textContent = name;
        button.append(spelling, caption);
        button.addEventListener('click', () => selectAndPlay(toneGroup(selection)[t]));
        recordings.appendChild(button);
        return { button, spelling };
    });

    function getFrame() {
        const mode = viewSelect.value;
        const { c, v, t } = selection;
        const index = { consonant: c, vowel: v, tone: t }[mode];
        const key = mode === 'compare' ? [mode, c, v].join(':') : [mode, index].join(':');
        if (frame?.key === key) return frame;

        if (mode === 'compare') {
            return { mode, key, items: toneGroup(selection), detail: consonantLabel(c) + ' + ' + vowels[v] };
        }
        if (mode === 'all') {
            return { mode, key, items: syllables, detail: 'All consonants, vowels, and tones' };
        }

        const axes = {
            consonant: ['c', consonantLabel(c)],
            vowel: ['v', vowels[v]],
            tone: ['t', toneLabel(t)]
        };
        const [coordinate, detail] = axes[mode];
        return { mode, key, items: syllables.filter(item => item[coordinate] === index), detail };
    }

    function refresh() {
        const { c, v, t, spelling } = selection;
        const { marker, name } = tones[t];
        consonantSelect.selectedIndex = c;
        vowelSelect.selectedIndex = v;
        toneSelect.selectedIndex = t;

        const parts = [consonants[c], vowels[v], marker].filter(Boolean);
        const description = spelling + ' = ' + parts.join(' + ') + ' · ' + name +
            (marker ? ' tone' : ' tone (no marker)');
        if (breakdown.textContent !== description) breakdown.textContent = description;

        const playing = playback?.items[playback.index];
        toneGroup(selection).forEach((item, i) => {
            const { button, spelling: label } = toneButtons[i];
            if (label.textContent !== item.spelling) {
                label.textContent = item.spelling;
                button.setAttribute('aria-label', 'Play ' + item.spelling + ', ' + tones[i].name + ' tone');
            }
            button.setAttribute('aria-current', String(selection === item));
            button.dataset.playing = String(playing === item);
        });

        if (playback?.repeatGroup) {
            playAll.textContent = 'Stop playback';
        } else {
            playAll.textContent = 'Play all seven';
        }
        const nextFrame = getFrame();
        if (frame !== nextFrame) {
            frame = nextFrame;
            viewSummary.textContent = count.format(frame.items.length) + ' combinations · ' + frame.detail;
        }
        chart?.update(frame, selection, playing);
    }

    function displaySearchStatus(message, error) {
        searchStatus.hidden = !message;
        searchStatus.textContent = message;
        searchStatus.dataset.error = String(error);
        search.setAttribute('aria-invalid', String(error));
    }

    function stopPlayback(message = '') {
        const clip = playback?.clip;
        playback = null;
        if (clip) {
            clip.onended = null;
            clip.onerror = null;
            clip.pause();
        }
        playbackStatus.textContent = message;
        refresh();
    }

    function playNext(request) {
        if (request !== playback) return;

        const { clip, index } = request;
        const item = request.items[index];
        if (!item) {
            stopPlayback('Finished playing all seven tones.');
            return;
        }

        clip.src = '_static/mp3/' + item.spelling + '.mp3';
        playbackStatus.textContent = 'Playing ' + item.spelling +
            (request.repeatGroup ? ' (' + (index + 1) + ' of ' + tones.length + ').' : '.');
        refresh();

        const reportError = () => {
            if (playback === request && request.index === index) {
                stopPlayback('Could not play ' + item.spelling + '. Try again.');
            }
        };
        clip.onerror = reportError;
        clip.onended = () => {
            if (playback !== request) return;
            if (request.repeatGroup) {
                request.index += 1;
                playNext(request);
            } else {
                stopPlayback();
            }
        };
        clip.play().catch(reportError);
    }

    function startPlayback(items, repeatGroup = false) {
        stopPlayback();
        playback = { items, repeatGroup, index: 0, clip: new Audio() };
        playNext(playback);
    }

    function selectAndPlay(item) {
        selection = item;
        displaySearchStatus('', false);
        startPlayback([item]);
    }

    search.form.addEventListener('submit', event => {
        event.preventDefault();
        const spelling = search.value.trim().toLowerCase();
        const item = bySpelling.get(spelling);
        if (!item) {
            displaySearchStatus('No matching syllable. Try po, poj, or cag.', true);
            return;
        }

        selection = item;
        viewSelect.value = 'compare';
        stopPlayback();
        search.value = item.spelling;
        displaySearchStatus('Found ' + item.spelling + '.', false);
        chart?.reset();
    });

    search.addEventListener('input', () => displaySearchStatus('', false));
    [consonantSelect, vowelSelect, toneSelect].forEach(select => {
        select.addEventListener('change', () => {
            const offset = (consonantSelect.selectedIndex * vowels.length + vowelSelect.selectedIndex) * tones.length;
            selection = syllables[offset + toneSelect.selectedIndex];
            displaySearchStatus('', false);
            stopPlayback();
        });
    });
    viewSelect.addEventListener('change', refresh);
    playAll.addEventListener('click', () => {
        if (playback?.repeatGroup) {
            stopPlayback('Playback stopped.');
        } else {
            startPlayback(toneGroup(selection), true);
        }
    });
    resetView.addEventListener('click', () => chart?.reset());

    function loadScript(url) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = url;
            script.addEventListener('load', resolve, { once: true });
            script.addEventListener('error', reject, { once: true });
            document.head.appendChild(script);
        });
    }

    function createChart(THREE, onPick) {
        const element = get('scene');
        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        const canvas = renderer.domElement;
        element.appendChild(canvas);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
        const controls = new THREE.OrbitControls(camera, canvas);
        const labels = document.createElement('div');
        labels.className = 'rpa-explorer__labels';
        element.appendChild(labels);

        // The three scales give each complete axis a comparable length.
        const positions = new Map(syllables.map(item => [
            item, new THREE.Vector3(item.v * 1.35, item.c * 0.30, item.t * 2.4)
        ]));
        const axisDefinitions = [
            { coordinate: 'x', color: 0x99e6ac },
            { coordinate: 'y', color: 0xffb786 },
            { coordinate: 'z', color: 0xa9c9ff }
        ];

        const dot = document.createElement('canvas');
        dot.width = dot.height = 32;
        const context = dot.getContext('2d');
        context.fillStyle = '#fff';
        context.beginPath();
        context.arc(16, 16, 14, 0, Math.PI * 2);
        context.fill();
        const dotTexture = new THREE.CanvasTexture(dot);
        const pointStyle = {
            map: dotTexture, transparent: true, alphaTest: 0.3,
            sizeAttenuation: false, depthWrite: false
        };

        const points = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({
            ...pointStyle, color: 0xbacbe3
        }));
        const focus = new THREE.Points(
            new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3)),
            new THREE.PointsMaterial({ ...pointStyle, color: 0xffcc74, size: 15, depthTest: false })
        );
        focus.renderOrder = 1;
        scene.add(points, focus);

        const bounds = new THREE.Box3();
        const box = new THREE.Box3Helper(bounds, 0x547096);
        box.material.transparent = true;
        box.material.opacity = 0.4;
        const axesGeometry = new THREE.BufferGeometry();
        const axes = new THREE.LineSegments(axesGeometry, new THREE.LineBasicMaterial({ vertexColors: true }));
        scene.add(box, axes);

        const projected = new THREE.Vector3();
        let frame;
        let labelPositions = [];
        let width;
        let height;
        let queued = false;
        let pointerStart;

        function placeLabel(label, point, offset) {
            projected.copy(point).project(camera);
            label.hidden = projected.z < -1 || projected.z > 1 ||
                Math.abs(projected.x) > 0.93 || Math.abs(projected.y) > 0.91;
            label.style.left = ((projected.x + 1) * width / 2) + 'px';
            label.style.top = ((1 - projected.y) * height / 2 + offset) + 'px';
        }

        function draw() {
            if (queued) return;
            queued = true;
            requestAnimationFrame(() => {
                queued = false;
                renderer.render(scene, camera);
                labelPositions.forEach(({ label, point }) => placeLabel(label, point, -8));
            });
        }

        function resize() {
            const nextWidth = Math.max(element.clientWidth, 1);
            const nextHeight = Math.max(element.clientHeight, 1);
            if (width === nextWidth && height === nextHeight) return;
            width = nextWidth;
            height = nextHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
            reset();
            draw();
        }

        function reset() {
            if (!frame) return;
            const center = bounds.getCenter(new THREE.Vector3());
            const radius = bounds.getSize(new THREE.Vector3()).length() / 2;
            const padding = frame.mode === 'compare' ? 0.9 : 1.2;
            const distance = radius / (Math.tan(Math.PI / 9) * Math.min(camera.aspect, 1)) * padding;
            controls.target.copy(center);
            controls.minDistance = 1.2;
            controls.maxDistance = Math.max(distance * 3, 30);
            camera.position.copy(center).add(
                new THREE.Vector3(-1.2, 0.82, 0.85).normalize().multiplyScalar(distance)
            );
            controls.update();
            draw();
        }

        function drawFrame(nextFrame) {
            const vectors = nextFrame.items.map(item => positions.get(item));
            const coordinates = vectors.flatMap(vector => vector.toArray());
            points.geometry.dispose();
            points.geometry = new THREE.BufferGeometry().setAttribute(
                'position', new THREE.Float32BufferAttribute(coordinates, 3)
            );
            points.material.size = nextFrame.mode === 'compare' ? 11 : nextFrame.mode === 'all' ? 4 : 6;

            bounds.setFromPoints(vectors);
            axisDefinitions.forEach(({ coordinate }) => {
                const span = bounds.max[coordinate] - bounds.min[coordinate];
                const pad = span === 0 ? 1.35 : 0.7;
                bounds.min[coordinate] -= pad;
                bounds.max[coordinate] += pad;
            });

            const endpoints = [];
            const colors = [];
            axisDefinitions.forEach(({ coordinate, color }) => {
                const end = bounds.min.clone();
                end[coordinate] = bounds.max[coordinate];
                endpoints.push(...bounds.min.toArray(), ...end.toArray());
                const rgb = new THREE.Color(color).toArray();
                colors.push(...rgb, ...rgb);
            });
            axesGeometry.dispose();
            axesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(endpoints, 3));
            axesGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            axesGeometry.computeBoundingSphere();
        }

        function addLabel(text, point) {
            const label = document.createElement('span');
            label.className = 'rpa-explorer__label';
            label.textContent = text;
            labels.appendChild(label);
            return { label, point };
        }

        function update(nextFrame, selected, playing) {
            const changedFrame = frame !== nextFrame;
            frame = nextFrame;
            if (changedFrame) drawFrame(frame);

            const highlighted = frame.items.includes(playing) ? playing : selected;
            focus.position.copy(positions.get(highlighted));
            labels.replaceChildren();
            labelPositions = (frame.mode === 'compare' ? frame.items : [highlighted]).map(item => {
                const annotation = addLabel(item.spelling, positions.get(item));
                annotation.label.dataset.selected = String(item === highlighted);
                return annotation;
            });
            if (changedFrame) reset();
            draw();
        }

        canvas.addEventListener('pointerdown', event => {
            pointerStart = { x: event.clientX, y: event.clientY };
        });

        canvas.addEventListener('click', event => {
            if (!frame || !pointerStart ||
                Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 6) return;

            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            let nearest;
            let distance = 13;
            let depth = Infinity;
            frame.items.forEach(item => {
                projected.copy(positions.get(item)).project(camera);
                if (Math.abs(projected.z) > 1) return;
                const difference = Math.hypot(
                    x - (projected.x + 1) * width / 2,
                    y - (1 - projected.y) * height / 2
                );
                if (difference < distance || (difference === distance && projected.z < depth)) {
                    nearest = item;
                    distance = difference;
                    depth = projected.z;
                }
            });
            if (nearest) onPick(nearest);
        });

        controls.addEventListener('change', draw);
        new ResizeObserver(resize).observe(element);
        resize();
        return { update, reset };
    }

    refresh();
    explorer.hidden = false;

    const threeBase = 'https://cdn.jsdelivr.net/npm/three@0.124.0/';
    loadScript(threeBase + 'build/three.min.js')
        .then(() => loadScript(threeBase + 'examples/js/controls/OrbitControls.js'))
        .then(() => {
            chart = createChart(window.THREE, selectAndPlay);
            chart.update(frame, selection, playback?.items[playback.index]);
            get('axes').hidden = false;
            sceneStatus.hidden = true;
            resetView.disabled = false;
        })
        .catch(() => {
            sceneStatus.textContent = 'The 3D view is unavailable. You can still search and listen above.';
            stage.querySelector('canvas')?.remove();
        });
})();
