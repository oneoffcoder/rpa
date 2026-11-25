.. _visualization:

Syllable Explorer
=================

Choose a consonant and vowel, then listen to how each of the seven tones changes the syllable.

.. raw:: html

    <noscript><p>Enable JavaScript to use the explorer, or <a href="syllables.html">browse the syllable tables</a>.</p></noscript>
    <div class="rpa-explorer" id="rpa-explorer" hidden>
        <form class="rpa-explorer__search" role="search" aria-label="Syllable search">
            <label for="rpa-search">Find a syllable</label>
            <div class="rpa-explorer__search-row">
                <input id="rpa-search" type="search" placeholder="e.g. cag" autocomplete="off"
                       autocapitalize="none" spellcheck="false" aria-describedby="rpa-search-status" required>
                <button type="submit">Find</button>
            </div>
            <p id="rpa-search-status" role="status" hidden></p>
        </form>

        <div class="rpa-explorer__filters">
            <div class="rpa-explorer__consonant"><label for="rpa-consonant">Consonant</label><select id="rpa-consonant"></select></div>
            <div class="rpa-explorer__vowel"><label for="rpa-vowel">Vowel</label><select id="rpa-vowel"></select></div>
            <div class="rpa-explorer__tone"><label for="rpa-tone">Tone</label><select id="rpa-tone"></select></div>
        </div>

        <section class="rpa-explorer__listening" aria-labelledby="rpa-listen-title">
            <div class="rpa-explorer__heading">
                <div>
                    <h2 id="rpa-listen-title">Listen to the seven tones</h2>
                    <p id="rpa-breakdown" aria-live="polite"></p>
                </div>
                <button type="button" id="rpa-play-all">Play all seven</button>
            </div>
            <div class="rpa-explorer__recordings" id="rpa-recordings" role="group" aria-label="Tone recordings"></div>
            <p class="rpa-explorer__playback" id="rpa-playback" role="status"></p>
        </section>

        <section aria-labelledby="rpa-scene-title">
            <div class="rpa-explorer__heading">
                <div>
                    <h2 id="rpa-scene-title">Explore in 3D</h2>
                    <p id="rpa-view-summary" aria-live="polite"></p>
                </div>
                <div class="rpa-explorer__view-controls">
                    <label for="rpa-view">View</label>
                    <select id="rpa-view">
                        <option value="compare">Compare tones</option>
                        <option value="consonant">Hold consonant</option>
                        <option value="vowel">Hold vowel</option>
                        <option value="tone">Hold tone</option>
                        <option value="all">Full cube</option>
                    </select>
                    <button type="button" id="rpa-reset-view" disabled>Reset view</button>
                </div>
            </div>
            <div class="rpa-explorer__stage" id="rpa-stage">
                <div class="rpa-explorer__canvas" id="rpa-scene" aria-hidden="true"></div>
                <ul class="rpa-explorer__axes" id="rpa-axes" aria-label="Chart axes" hidden>
                    <li class="rpa-explorer__consonant">Consonant</li>
                    <li class="rpa-explorer__vowel">Vowel</li>
                    <li class="rpa-explorer__tone">Tone</li>
                </ul>
                <p class="rpa-explorer__scene-status" id="rpa-scene-status" role="status">Loading 3D view…</p>
            </div>
            <p class="rpa-explorer__instructions">Drag to rotate, scroll or pinch to zoom, and select a point to hear its syllable.</p>
        </section>
    </div>
    <script defer src="_static/js/viz.js"></script>
