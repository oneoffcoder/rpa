CVT
===

.. blockdiag::

   diagram {
      default_shape = roundedbox
      span_width = 32
      span_height = 20
      default_fontsize = 11
      edge_layout = normal
      orientation = landscape

      C [label = "c", color = "#f55f36", shape = circle, textcolor = white]
      V [label = "v", color = "#15d649", shape = circle, textcolor = white]
      T [label = "t", color = "#365cf5", shape = circle, textcolor = white]
      S [label = "Syllable", color = "#e0e0e0"]

      C, V, T -> S
   }

The mental model of RPA is with understanding the so-called ``CVT`` (Consonant, Vowel, Tone) composition of a ``syllable``. For all intents and purposes, a syllable is simply a sound. Hmong is a language, like Mandarin, that is largely mono-syllabic; meaning, one sound is one word. Of course, there are multi-syllable words in Hmong too, but, these words themselves are composed of the elementary mono-syllables. After you learn the mono-syllables, this achievement paves the way to learning compound words in Hmong. 

Back to ``CVT``. Simply stated, a ``consonant`` is the beggining of a word, a ``vowel`` is the middle of a word and a ``tone`` is the ending of a word. For more technical articles of these concepts, see the following `constant <https://en.wikipedia.org/wiki/Consonant>`_, `vowel <https://en.wikipedia.org/wiki/Vowel>`_ and `tone <https://en.wikipedia.org/wiki/Tone_(linguistics)>`_ articles. In RPA, there are only so many consonants, vowels and tones. In fact, there are 

* 56 consonants,
* 13 vowels, and
* 7 tones.

Mathematically, there are 5,096 (:math:`56 \times 13 \times 7`) possible permutations of syllables that may be created using these finite sets of consonants, vowels and tones.

.. note::

    RPA's ``CVT`` approach to creating syllables is not perfect. There are sounds that cannot be spelled using the CVT approach. Such sounds include those used in rituals and/or religious activities.

However, some words do not have a consonant. Meaning, we have 91 (:math:`13 \times 7`) possible syllables without a consonant. In all, there are 5,187 (:math:`5,096 + 91`) total possible syllables. 

Going further, some words are just vowels. So, actually, there are 5,200 (:math:`5,187 + 13`) total possible syllables in RPA.

.. note::

    The ``CVT`` mental model of Hmong syllables and/or words is so powerful that many other Hmong writing systems are modeled using this approach. For example, the famous ``Phaj Hauj`` script uses ``TCV`` (the tone comes first) to model writing syllables.
    
Let's dive into the consonants, vowels and tones.

Consonants
----------

Consonants, again, are the beggining of a sound (or word). In RPA, it is convenient to parition consonants into the number of letters that a consonants have.

* c, d, f, h, k, l, m, n, p, q, r, s, t, v, x, y, z
* ch, dh, hl, hm, hn, kh, ml, nc, nk, np, nq, nr, nt, ny, ph, pl, qh, rh, th, ts, tx, xy
* hml, hny, nch, nkh, nph, npl, nqh, nrh, nth, nts, ntx, plh, tsh, txh
* nplh, ntsh, ntxh

In RPA, consonants are called ``tsiaj ntawv txiv``. The families of consonants and their English and Hmong names are shown below.

.. csv-table::
    :header: Number of Letters, English, Hmong

    1, ``mono-consonants``, ``tsiaj ntawv txiv tab``
    2, ``bi-consonants``, ``tsiaj ntawv txiv txooj``
    3, ``tri-consonants``, ``tsiaj ntawv txiv cov peb tug``
    4, ``quadra-consonants``, ``tsiaj ntawv txiv cov plaub tug``

Vowels
------

Just as in Mandarin and English, all words have vowels. In RPA, all syllables (or words) have one of the following vowels. Again, we group them into the number of letters that composes the vowels and list them alphabetically.

* a, e, i, o, u, w
* ai, au, aw, ee, ia, oo, ua

In RPA, vowels are called ``tsiaj ntawv niam``.

.. csv-table::
    :header: Number of Letters, English, Hmong

    1, ``mono-vowels``, ``tsiaj ntawv niam tab``
    2, ``bi-vowels``, ``tsiaj ntawv niam txooj``

Tones
-----

There are 7 tones in RPA. The Hmong language is sometimes called the `tonal champion` of East Asian languages for having so many tones. Compare the Hmong language with 7 tones to Mandarin, which has 4 tones. The tones are as follows.

.. csv-table::
    :header: Tone, Marker, Example

    High falling, j, poj
    Low, s, pos
    Mid rising, v, pov
    Creaky, m, pom
    Low falling breathy, g, pog
    High, b, pob
    Mid, _, po
    
.. warning::

    The tone represented by the underscore ``_`` is really blank, but denoted here with an underscore to make it explicit. The underscore is never used to represent a tone and will be omitted later; the context will make it clear.

.. note::

    Sometimes ``d`` is also used as a tone!

In RPA, tones are called ``tsiaj ntawv cim``.