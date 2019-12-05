Grammar
=======

Pronouns
--------

Here are the basic pronouns in Hmong.

.. csv-table::
    :header: Point of View, English, Single, Double, Multiple

    First Person, Me, Kuv, Wb, Peb
    Second Person, You, Koj, Neb, Nej
    Third Person, Him/Her, Nws, Nkawv, Lawv

Classifiers
-----------

Classifiers are basically ``metrics``. For example, in the phrase, ``an ounce of gold``, the word ``ounce`` is a metric. Here are some Hmong classifiers.

.. csv-table::
    :header: Classifier, Application

    rab, "long, straight object; masculine"
    lub, "round, circular object; feminine"
    txoj, string or line like
    tsob, plant or bush
    leeg, living thing (self-referential)
    phau, book
    daim, sheet or flat object
    tug, generic for living things
    zaj, "song, poem, instrumental"
    qho, generic classifier with non-trivial usage

.. note::

    The ``zaj`` classifier is used to refer to a lot of artistic work. The word ``zaj`` means ``dragon`` in English. In Hmong creation stories, it is said that Hmong people learned of art, culture and technology from the Old Dragon, ``Zaj Laug``. Hence, many things are classified as ``zaj``.

    * zaj nkauj : song
    * zaj qeej: reed pipe instrumental
    * zaj tshoob: wedding ceremonial song
    * zaj dab neeg: Hmong pre-history story
    * zaj txiv xaiv: proverbial funeral ritual song

.. note::

    The ``qho`` or ``qhov`` classifier has non-trivial usage. In the simplest case, it can be used for hole-like things such as door ``qhov rooj``, eye ``qhov muag`` or ground holes ``qhov av``. In other cases, you can denote generic items in a list such as first item ``qhov ib``, second item ``qhov ob`` and so on.

Sentence structures
-------------------

The best way to understand sentence structures is just to observe and study a variety of sentences. Here's a few examples.

.. warning::

    `Part-of-Speech <https://en.wikipedia.org/wiki/Part-of-speech_tagging>`_ ``POS`` tagging is a very difficult `machine learning <https://en.wikipedia.org/wiki/Machine_learning>`_ and `natural language processing <https://en.wikipedia.org/wiki/Natural_language_processing>`_ ``NLP`` task. There may be mistakes in tagging the parts of speech in these examples. As a human task, POS tagging is not any easier.

Basic
^^^^^

The basic sentence structure in Hmong is like English. 

.. csv-table::

    subject, verb, object
    Kuv, haus, dej
    I, drink, water

Time
^^^^

What about time? Time may be placed at the start or end of a sentence. Noticed that the time specifies the verb tense (past, present or future).

.. csv-table::

    time, subject, verb, object
    Nag hmo, kuv, haus, dej
    Yesterday, I, drank, water

.. csv-table::

    subject, verb, object, time
    Kuv, haus, dej, nag hmo
    I, drank, water, yesterday

Here's how time may change the tense of a verb to future.

.. csv-table::

    Tag kis, kuv, haus, dej
    Tomorrow, I, will drink, water

Drop the time to imply present tense.

.. csv-table::

    subject, verb, object
    Kuv, haus, dej
    I, am drinking, water

The auxillary word ``lawm`` is identical to the Mandarin `le <https://www.digmandarin.com/use-le-in-chinese.html>`_. When used, it may also signal past tense.

.. csv-table::

    pronoun, verb, object, auxillary
    Kuv, haus, dej, lawm
    I, drank, water, already

Here are some more `grammatical aspects <https://en.wikipedia.org/wiki/Grammatical_aspect>`_ that modifies the verb over time.

.. csv-table::

    Kuv, ``tab tom``, haus, dej
    I, just started, to drink, water

.. csv-table::

    Kuv, ``yuav``, haus, dej
    I, am going to, drink, water

.. csv-table::

    Kuv, ``mam li``, haus, dej
    I, will then, drink, water

Adjectives
^^^^^^^^^^

Typically, unlike English, adjectives come after nouns. The sentence below translate into English as ``I like red cars.`` Note also that singular or plural (for cars in this case) is derived from the context. To be clear, you may add quantifiers and modifiers.

.. csv-table::

    pronoun, verb, noun, adjective
    Kuv, nyiam, cheb, liab
    I, like, cars, red

The following sentence uses a quantity and classifier to specify singularity with cars. The English translation is ``I like a red car.``

.. csv-table::

    pronoun, verb, quantity, classifier, noun, adjective
    Kuv, nyiam, ib, lub, cheb, liab
    I, like, one, <class>, car, red

Sometimes, adjectives may also come before nouns, and the meaning is very different. The following sentence translates to ``He is a good seed.`` Note that the word ``nws`` has no implied gender (it could be ``he`` or ``she``), although we simply imputed ``he``.

.. csv-table::

    pronoun, verb, adjective, noun
    Nws, yog, zoo, noob
    He, is, good, seed

Now, swap the adjective and noun positions. The following sentence translates to ``He is from a good family.``

.. csv-table::

    pronoun, verb, noun, adjective
    Nws, yog, noob, zoo
    He, is, seed, good

If you are a ``noob zoo``, it does not necessarily mean you are a good person, just that you are from a good family. If you are a ``zoo noob``, it means you are a good person, but not necessarily from a good family.

Adverbs
^^^^^^^

How about adverbs? The following sentence translates to ``I like to eat shrimp a lot.``

.. csv-table::

    pronoun, verb, verb, noun, adverb
    Kuv, nyiam, noj, cws, heev
    I, like, to eat, shrimp, much

The following sentence translates to ``I like to eat jumbo shrimps a lot.`` The adverb ``heev`` modifies the verb ``nyiam``, and is quite a distance away from it. 

.. csv-table::

    pronoun, verb, verb, noun, adverb, adverb
    Kuv, nyiam, noj, cws, loj, heev
    I, like, to eat, shrimp, big, much

Classifiers
^^^^^^^^^^^

Let's have fun with classifiers. The following sentences translates to ``His ambition is very big.`` However, notice how we change the classifier from ``lub`` to ``rab``? The ``lub`` classifier is feminine, and the ``rab`` classifier is masculine. Sometimes, these two classifiers may be swapped (they are not interchangeable, since they are not equal). The ambition ``lub peev xwm`` is different from the ambition ``rab peev xwm``. In this case, ``rab peev xwm`` is dominant over ``lub peev xwm``.

.. csv-table::

    pronoun, classifier, noun, adjective, adverb
    Nws, lub, peev xwm, loj, heev
    Nws, rab, peev xwm, loj, heev
    His, <class>, ambition, big, much

.. note::

    A woman may have ``rab peev xwm`` or ``lub peev xwm`` (and likewise for a man). The classifier may genderize the noun, but it does not constrain the associated pronoun.

Verbs in depth
^^^^^^^^^^^^^^

Verbs are well-behaved until you have to start describing how you wear clothes. In English, you 

* wear a hat,
* wear a scarf,
* wear gloves,
* wear shoes,
* wear glasses,
* wear a shirt,
* wear a pants, and
* wear shoes.

No matter what you ``wear`` in English, you ``wear`` it. In Hmong, verbs used denote wearing clothing changes depending on the body part or clothing particle. Here are the right verbs to use when referring to wearing clothing.

.. csv-table:: Wearing Verbs
    :header: Items, Hmong Verb

    hat, ntoo
    "watch, bracelet, necklace, glasses, earrings", coj
    "gloves, socks", looj
    "belt, backpack", sia
    "pants, shirt", hnav
    shoes, rau

Observe these verbs in action (no pun intended). Also, pay attention to the ``classifiers``.

.. csv-table:: I wear a hat

    Kuv, ntoo, ib, lub, kaus mom
    I, wear, a, <class>, hat

.. csv-table:: I wear two earrings

    Kuv, coj, ob, lub, qhwv ntsej
    I, wear, two, <class>, earrings

.. csv-table:: You wear two socks

    Koj, looj, ob, qho, thoom khwm
    You, wear, two, <class>, socks

.. csv-table:: You wear a belt

    Koj, sia, ib, txoj, siv tawv
    You, wear, a, <class>, belt

.. csv-table:: She wears a pair of pants

    Nws, hnav, ib, lub, ris
    She, wears, a, <class>, pants

.. csv-table:: They are wearing two shoes that are the same

    Nkawv, rau, ob, nkawg, khau, ib yam
    They, wear, two, <class>, shoes, the same

Questions
^^^^^^^^^

One way to ask questions is to simply present a choice and the negation of that choice. The word ``los`` means ``or`` in English; a longer form of ``los`` is ``los sis`` (and may be substituted).

.. csv-table:: Do you want to drink water?

    Koj, huas, los, tsis haus, dej, ?
    You, drink, or, not drink, water, ?

.. csv-table:: Do you want to go?

    Koj, mus, los, tsis mus, ?
    You, go, or, not go, ?

Another way to ask these questions is to use ``puas`` (before the verb) which means ``will you``.

.. csv-table:: Do you want to drink water?

    Koj, puas, haus, dej, ?
    You, will, drink, water, ?

.. csv-table:: Do you want to go?

    Koj, puas, mus, ?
    You, will, go, ?

The particles ``ma`` and ``ne`` may also be added to end of a sentence to make a question.

.. csv-table:: How much does it cost?

    Pes tsawg, ma, ?
    How much, <particle>, ?
    
.. csv-table:: And him?

    Nws, ne, ?
    Him, <particle>, ?

The `5W1H <https://en.wikipedia.org/wiki/Five_Ws>`_ are asked as follows.

.. csv-table::
    :header: English, Hmong
    
    Who?, Leej twg?
    What?, Dab tsi?
    When?, Thaum twg?
    Where?, Qhov twg?
    Why?, Vim li cas?
    How?, Ua li cas?