$(document).ready(function() {
    let lock = false;

    function acquireLock() {
        lock = true;
    }

    function releaseLock() {
        lock = false;
    }

    function getConsonants() {
        let C = 'c, d, f, h, k, l, m, n, p, q, r, s, t, v, x, y, z,';
        C += 'ch, dh, hl, hm, hn, kh, ml, nc, nk, np, nq, nr, nt, ph, pl, qh, rh, th, ts, tx, xy,';
        C += 'hml, hny, nch, nkh, nph, nqh, nrh, nth, nts, ntx, plh, tsh, txh,';
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
        let T = 'j, s, v, m, g, b, _'.split(',')
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
                T.forEach(t => {
                    const cvt = `${c}${v}${t}`.replace('_', '');
                    S.push(cvt)
                });
            });
        });

        V.forEach(v => {
            T.forEach(t => {
                const vt = `${v}${t}`.replace('_', '');
                S.push(vt);
            });
        });

        V.forEach(v => {
            S.push(v);
        });

        return S;
    }

    function modifyMp3Links() {
        const anchors = $('a[class="reference external"]')
            .filter(function(index) { 
                let href = $(this).attr('href');
                return href.endsWith('mp3');
            });
        $(anchors).each(function(index, value) {
            let href = $(this).attr('href');
            $(this).click(function() {
                console.log(href);
                var sound = new Howl({
                    src: [href]
                })
                sound.play();
                return false;
            });
        });
    }

    function modifyGenerateLinks(syllables) {
        const anchors = $('a[class="reference external"]')
            .filter(function() { 
                let href = $(this).attr('href');
                return href.startsWith('generate.html?q=');
            });
        
        $(anchors).each(function(index, value) {
            const href = $(this).attr('href');
            const sounds = href.substring('generate.html?q='.length)
                        .split('+')
                        .filter(v => syllables.includes(v))
                        .map(v => `_static/mp3/${v}.mp3`)
                        .map(v => new Audio(v));
            $(this).click(function() {
                if (lock === false && sounds.length > 0) {
                    acquireLock();
                    let i = -1;
                    play();
                    function play() {
                        i++;
                        if (i == sounds.length) {
                            releaseLock();
                            return;
                        }

                        try {
                            sounds[i].addEventListener('ended', play);
                        } catch(e) {
                            // swallow
                        }
                        
                        try {
                            sounds[i].play();
                        } catch(e) {
                            // swallow
                        }
                        
                    }
                }
                
                return false;
            });
        });
    }

    console.log('ready!');
    
    const syllables = getSyllables();

    modifyMp3Links();
    modifyGenerateLinks(syllables);
    
    console.log('done!');
});