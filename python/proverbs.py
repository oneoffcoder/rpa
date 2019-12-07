import itertools
import string

def get_syllables():
    C = """
    c, d, f, h, k, l, m, n, p, q, r, s, t, v, x, y, z,
    ch, dh, hl, hm, hn, kh, ml, nc, nk, np, nq, nr, nt, ny, ph, pl, qh, rh, th, ts, tx, xy,
    hml, hny, nch, nkh, nph, npl, nqh, nrh, nth, nts, ntx, plh, tsh, txh,
    nplh, ntsh, ntxh
    """.strip()

    V = """
    a, e, i, o, u, w,
    ai, au, aw, ee, ia, oo, ua
    """.strip()

    T = """
    j, s, v, m, g, b, _
    """.strip()

    C = [c.strip() for c in C.split(',')]
    V = [v.strip() for v in V.split(',')]
    T = [t.strip() for t in T.split(',')]

    CVT = [f'{c}{v}{t}' for c, v, t in itertools.product(C, V, T)]
    VT = [f'{v}{t}' for v, t in itertools.product(V, T)]

    syllables = []
    syllables.extend(CVT)
    syllables.extend(VT)
    syllables.extend(V)

    syllables = [s.replace('_', '') for s in syllables]

    return syllables

s = """
Cag ntoos qhia taus npub, hlua hneev qhia xib xub.
Cawv qab thawj txaum, qoob loo zoo thawj phaum.
Cem nom ces nom nplua, cem tus ruam ces ruam caum tua.
Cem yuav cem kom yog lus, hais lo lus yuav hais kom tus.
Cuaj leeg tub tsis cuag leej txi, cuaj lub hnub qub tsis cuag lub hli.
Cuaj lub cub tawg sov tsis cuag ib lub hnub; cuaj leeg ntxhais zoo tsis cuag ib leeg tub. 
Cuaj lub hnub qub tsis cuag ib lub hli; cuaj leeg tub tsis npaum ib leeg txi. 
Dab qhuas tsis zoo thiaj tso dab qus los ntxias, poj niam tub se tsis zoo, lauj kaub thiaj tsoo ntug yias.
Dab tau tua pa ncho, neej tau nqaij tau hno.
Dab tsev tsis zoo thiaj cia dab qus ntxias; nyab tsis zoo nyab tsev thiaj tsoo yias. 
Dag dab tau kev tuag, dag neeg tau kev txaj muag.
Deev cuag deev nkauj sua, noj cuag noj nqaij xa qhua.
Deev hluas rau kev dav, zoo nkaus li txhom ntses hauv qhov cav.
Deev hluas rau kev deb, zoo nkaus li txhom ntses hauv qhov zeb.
Dej ntws los vim muaj tug kwj, yoov tsaws los yog vim nqaij tsw lwj.
Dev tsem dev tsis tom, luag yuav tua koj luag tsis hom.
Dev yeem kaws txha, npua yeem lib a.
Hais ruab zoo nrov ruab tsuag, hais hauv tsev ncha saum luag.
Hais txog kev lom zem ces xav luag; hais txog kev txom nyem ces xav tuag. 
Hla dej yuav hle khau, tsiv teb txav chaws yuav hle hau.
Hlais nqaij ntshaw zoo riam, ua neeg nyob ntshaw zoo poj niam.
Hlub kwv tij txhob tseg leej npawg, hlub niam hlub txiv txhob tseg pog yawg.
Hlub poj niam txhob tseg kwv tij, hlub nkauj hlub muam txhob tseg yawm yij.
Hov riam, riam ntse, txhuag siav, siav ntev.
Hwm poj niam tub se thiaj tau kaus roos, hwm dab hwm qhua thiaj ua lub neej nyob laus nkoos.
Ib re nplej muaj tshaj peb lub npluag, ib tug nkauj muaj tshaj peb tuag hluas.
Ib ya taws ua tsis siav ib lauj kaub mov, ib pluaj ntuag ua tsis tau ib ntshuas xov.
Ib ya taws ua tsis siav ib lauj kaub zaub, ib pluaj ntuag ua tsis tau ib chaws ntaub.
Kab coob piv rau kas, qhov muag loj piv rau plas.
Kaum leej tsom, tsis cuag li yus qhov muag pom.
Kev fab thev khiav, neeg txom nyem thev siav.
Kev fab thev taug, neeg txom nyem thev laus.
Kev nkauj kev nraug hais luag nyav, los txhob kuam hooj haus ntshav.
Kev pluag tsis yog tim txoj hmoo, kev tshaib plab tsis yog tim qoob loo.
Kev ua noj, tsis tas yuav cia luag ua tus coj; kev ua hnav, tsis tas cia luag ua tus tsav.
Khiav nab ntsib qav; khiav dab ntsib tog cav; khiav tog cav zom pas av. 
Koj tug cia koj coj, thiaj nrog luag muaj noj; koj tug cia koj tsav, thiab muaj hnav.
Koob meej nrov cuag luag cav kaus ntxhw, mus txog hauv tsev tsis muaj chaw pw.
Koob meej nrov cuag luag cav pov haum, mus txog hauv tsev tsis muaj rooj zaum.
Kua txob ntsim yuav tau ntsev daw, poj niam phem yuav tau xuas ncaws taw.
Kws ntaus phom tua hneev, neeg dig muag txawj luj teev.
Lag luam zoo thawj ntsug, poj niam zoo thawj tug.
Lag nteg hnov lus ntxi, dig muag pom kev nruab qaig hli.
Lo zoo nco ib ntsis, lo phem nco tag ib sis.
Loj cuag haum vag, tom tsis yeej uab lag.
Lov hniav nyiam noj qhov phom, muaj hniav nyiam qhov tawv zom.
Luag hais cuaj lo, kaum lo, los tseem tsis cuag yus qhov muag pom tso.
Luag hais cuaj suab, kaum suab, los tseem tsis cuag yus txhais tes muab.
Luag pom muaj pom txaus, tsis cuag yus tau tuav nkaus.
Luag tsis yeem txhob muab luag pus, luag tsis kheev txiv txhob mus.
Luag txaj tsis nrog luag txaj, ces ua dev luam yaj; luag tsaw tsis nrog luag tsaw ces kiav hniag cawv caw lawg.
Luj tuag tu noob, tshuav tug nyob qab roob; nas tuag tu tsav, tshua tus nyob qab hav.
Lus ntxhov cuag pos kaus ntsaj, plaub ntxhov cuag plaub yaj.
Mam noj mam ntsaig, mam xav mam hais.
Me nyuam yaus ntshaw qe, laus neeg ntshaw pe.
Mov kuam qab tsis cuag mov nplej, cov lawv qab zoo tsis cuag cov ua ntej.
Muab tsis thov, luag tias ua tub sab; muaj npe tsis muaj cev, luag tias yog dab. 
Muaj cai tsis nyob lig, muaj tus saus yuav dig.
Muaj ib hub nyiaj txhob twm ob txiv tub ntsuag; muaj ib txhab qoo txhob twm ib tug nas nchuaj.
Muaj mob thiaj nrhiav tshuaj; muaj tuag thiaj nyiaj quaj. 
Muaj noj yuav faib qhua, muaj hnav yuav sib faib npua.
Muaj npe tsis muaj koob, yam muaj qhov noj tsis muaj cov noob.
Muaj npe tsis muaj lub cev, luag hais tias yog dab, muab tsis hais ua nte luag tias ua tub sab.
Muaj nyiaj, qhia kom tsim txiaj; muaj qoob loo, qhia kom tau zoo.
Muaj qoob loo noj thawm niaj, luag thiaj qhuas tias neeg tsim txiaj.
Muaj qoob loo noj thawm xyoo, luag thiaj qhuas tias yog neeg muaj hmoo.
Muaj tsawg nyuaj tus faib, muaj ntau nyuab tus saib.
Muaj zoo kwv tij sib hlub, tseem tsis cuag li muaj me nyuam zaum dub vog ncig ntug cub.
Nag los peb kob ces av ua kwj, dais tuag peb hnub ces dais lwj.
Nas tias nas nyob qhov av, plis kuj haig tias plis tau chaw dav.
Ncauj ntau tau kua haus, ua phem raug luag ntaus.
Neeg ntse hais qhov yog, neeg ruam tsis khib zog.
Neeg ntse hais yam tseeb, neeg ruam tsis khib xeeb.
Nej yaum cag cub, lawv yaum cag cub, lawv twb yuav teg nkag nplog niam lub.
Niam ncoo txiv puab, txiv ncoo niam pu; niam mloog txiv qhuab, txiv mloog niam hu.
Niam tshuab ntuag, txiv qaiv ntxaiv, niam txhob ncauj ntau txiv txhob nkawg lus xaiv.
Niam txiv sib ceg tas tsis nrauj txaj, kwv tij sib ceg tag tsis nrauj dab.
Nkauj nyob ces sov zos, nkauj ua nyab tag ces tsov los.
Noj ib pluag tas xav mus ploj, tsis muaj nom muaj tswv, tsiv muaj neeg xav coj, noj ib pluag tag xav mus ntua, tsis muaj nom muaj tswv, luag tsis xav yuav.
Noj mov qab vim muaj aub ncaug, tsis muaj xaiv muaj lus vim txawj hais sib raug.
Noj nceb yuav taug cav, yuav poj niam yuav nug neej nug tsav.
Noj nqaib yuav rawg muab, ua neej yuav cia luag qhuab.
Noj nqaij yuav rau ntsev, ua neeg nyog yuav tswm rub lub siab kom ntev.
Noj nqaij yuav rawg tais, ua neeg yuav tau cia luag hais.
Noj qa tsis puv plab, nqhis luag qoob tuam txhab.
Noj qav yuav rau ntsev, uaj neeg yuav siab ntev; noj qav yuav rau roj, ua neeg yuav siab loj. 
Noj txiv ntoo yuav saib noob, yuav poj tub se saib caj hmoob.
Noj zaub yuav rau roj, ua neeg nyob yuav tsum ua siab loj.
Nom siav dav hwm pav tim xyoob; nom siav hlob hwm kav toob. 
Nom zoo ces hlub pej xeem; nom tswv tsis zoo ces hlub pej neem. 
Npav tswv yim yuav nrhiav cov laus, npav sib ncaws yuav nrhiav me nyuam yaus.
Nplij kaus thiaj tau kawv kwv, nplij tub nplij roog thiaj tau zoo tshwv.
Nplij niam nplij txiv thiaj tau ntuj ntoo, nplij tub nplij roog thiaj tau zoo.
Nplooj yoog kab, noob yoo tsav.
Npua yeem noj xua, dev yeem noj qua.
Nqaij qab txha txiv ntoo qab plhaub, ua neeg nyob txob nta thiaj paub.
Nqhis nqaij khawb nas kos, tshaib plab khawb qos.
Nrhiav tshuaj nrhiav thaum muaj mob, nrhiav nas tsuag nrhiav thaum mis tsis nyob.
Nrog cov laus dab tsis hem, nrog cov nom yuav tsis cem.
Nrog cov laus dab tsis hem; nrog cov nom luag tsis cem. 
Ntau tus ua cuav poov xab, ntau txiv muam ua puas dab.
Nte taws haus tsev tsis sov, noj mov hauj luag tsi txau.
Ntiaj teb ceeb tsheej nyob nruab siab; ntuj ceeb tsheej nyob ntuj tsheej kiab. 
Ntoo laus ntoo khoob, neeg laus neeg ua tem toob.
Ntshai zeej xeeb tsis pheej, thiaj tsis teev tsheej.
Ntshaw kev muaj noj, yuav tsum rau siab ntso rau lub luag nroj.
Ntshe tim fab tsis pheej, thiaj tsim thawj zeej.
Ntsuag yim laus yim iab, niam txiv yim laus yim mob kiab.
Ntuj dub ntuj ntiab, lam ua neej xwb tsis paub leej twg lub siab.
Ntuj siab laj ti ntsua, ua neej nyob, xav muaj xwb tsis xav pluag
Ntuj siab laj ti ntsuas, ua neej nyob, paub hnub nyob tsis paub hnub tuag.
Nyiaj txiag nyob tsiv kev, txiv ntoo nyob chaw siab, tsoo tsuj tsoos npuag nyob nruab kiab.
Nyob deb txo dej txo cawv, nyob ze rho hluav taws hlawv.
Nyuj kub tiv nyuj tawv, nroj tsuag tuag cuag hlawv.
Nyuj ntshuag tsis tuag, nyuj ntshuag hlob tiav pwj, tub ntshuag tsis tuag tub ntshuag hlob ciaj xeev txwj.
Nyuj ua nyuj thauj, nees ua nees ris; dev daj ua dev daj thauj, dev dub ua dev dub ris.
Nyuj ua rau nyuj thauj, nees ua rau nees ris.
Paj tsis tawg txiv tsis txi, nkauj tsis rog, raug tsis ntsi.
Paub hnub muaj tsis paub hnub pluag, paub hnub nyob tsis paub hnub tuag.
Peb suab qhuas zoo dua saub kho, peb tug tes zoo dua ib tug ko.
Peb suab qhuas zoo dua saub saib, peb lub qe ntxias tsis cuag ib lub qe qaib.
Plaub hau ntxhov thiaj yuav zuag los ntsis; plaub ntug ntxhov thiaj yuav txwj laus los lis. 
Plaub hau ntxhov thiaj yuav zuag ntsis, plaub ntug ntxhov thiaj yuav txwj laug los lis.
Pluag ces pheej khwv los tsis muaj chaw tau, muaj ces tsis siv tsis muaj chaw rau.
Pob txha muaj chaw to; niam txiv muaj chaw nco. 
Poj niam khib xeeb, cem tus txiv piv rau neeg loj leeb.
Poj niam ruam, cem tus txiv piv rau neeg kaub huam.
Poj niam siab zoo los nrog tus txiv ua neej ncoo pu, poj niam siab phem los nrog tus txiv ua neej tsoo yias tsoo tsu.
Pom qhov zoo ces muab tes taum, pom qhov phem ces ho yuav nkawm.
Pom tsov yauv tuag; pom nom yuav plaug. 
Pom txiv qaub tsis los aub ncaug ces luag tiag yuav tuag, pom nyiaj txiag tsis nrog luag ntshaw ceg yog yuav sawm pluag.
Puag luag tub tsaws dej dag, deev luag poj tuaj ruab txag.
Puag luag tub tsaws dej ntab, nyiag luag qoob tuaj nruab txhab.
Qhiav laus qhiav ntsim, tus laus hais lus thiaj tej lim.
Qhiav laus qhiav ntsim, txus laus txus muaj tswv yim.
Qhiav laus qhiav quav; tus laus hais tus hluas yuav tsum yuav.
Qiv luag tais rau tshais, khaws luag lus los hais.
Qiv luag toj yug yaj, qiv luag ncauj los hais zaj.
Qoob tsis zoo tsuas plam nyog ib cim; poj tsis zoo plam tas ib sim. 
Qov qav poob kwj deg, qog neeg ceg tawv ces lov ceg.
Quab me qhia twm; kev tub nkeeg qhia huaj lwm. 
Quab qus muab ua tsis nyog qaib dib, zaub mos ua tsis nyog zaub tsib.
Siab loj yog tus coj, siab me ces yog qhev.
Siab ntev thiaj tau nom ua, zoo nraug thiaj tau nkauj sua.
Siab zuag cuag nplooj dos, siab dawb cuag plawv qos.
Siab zuag cuag nplooj nqeeb, siab dawb cuag paj yeeb.
Sib qhia thiaj txawj ntse, sib pab thiaj tsis ciaj luag qhev.
Suab neeg nrov dua sob nroo, lus phem nrov dua lus zoo.
Suab neeg nrov dua suab xob, lus phem ntsim dua kua txob.
Tau nom tswv zoo tam tau kaus kwv, tau nom tswv phem tam pab tsiaj tsis muaj tswv.
Tau no ces qhuas tias luag siab zoo, tsis tau noj ces thuam luag mus thawm xyoo.
Tau noj ceg qhuas tias luag siab tiaj, tsis tau noj ces thuam luag mus thawm niaj.
Tau phem nyab, ces tam ntua li ntes raub ris hauv vab.
Tau poj tub se nquag, ces ua lub neej nyob xws luag.
Tau poj tub se zoo, ces ua lub neej nyob nto moo.
Tau poj tub se zoo, zoo siab tam tau txhab qoob los noj thawm xyoo.
Tau zoo thaum yau, laus los kev tsom nyem raws tau.
Taum lag taum lig ua quav hma, tub lag tub lig ua luag puab ntsa.
Taum lag taum lig ua quav ntsooj, tub lag tub lig ua luag puab rooj.
Tawv cav los niam qhuas nceb txhais, nkauj laug los niam qhuas tias yob me nyuam ntxhais.
Tawv nkaus tsiv rauv, neeg laus tsiv yug.
Thaum luag tsaj tsis nrog luag tsaj, luag tsaj tas ces ua dev nuam yaj.
Thaum mo liab niam thiab txiv tos nrw tub tsis txawj loj, thaum tub hlob los tub tos nrhw niam thiab txiv tsis txawj ploj.
Thaum mos mas mos cuag ntaub tsuj, thaum ntxhib mas ntxhib cuag plab nyuj.
Thaum muaj ces muab noj ntau, thaum tas ces ua tsov tshaib tsov tsau.
Thaum pib ua yuav tsum tuav xam, ua tau los thiaj li tsis plam.
Thaum siav tu ces luag tias tuag; thaum nyiaj tu ces luag tias pluag. 
Thaum siav tus ces luag tias tuag, thaum nyiaj tu ces luag tias pluag.
Thaum zoo ces hais tias "mog mog", thaum phem ces cuag dev sib tog.
Thaum zoo siab ces sib tawb, thaum tsiv siab tuaj ces ua miv sib khawb.
Tij viv tam niam txiv, niam txiv tam rooj ntug.
Toj roob hauv pes quas ces quas ib ntsis, av luaj quas ces quas tas ib sis.
Tshaib mov khawb qos, tshaib nqaij khawb nas kos.
Tsim tub ntsoj ces khaum ploj; tsim tub ntsug khaum tuag. 
Tsis lees los tsis dim, lam lees lub yim.
Tsis lees los tsis yeej, lam lees lub txheej.
Tsis muaj nyiaj ces luag tias pluag, tsis muaj niam muaj txiv ces luag tias tub ntsuag.
Tsis muaj txooj qoob ces luag tias hav nroj, tsis muaj niam muaj txiv ces luag tias tub ntsoj.
Tsis muaj zoo rooj los lam nyob, tsis muaj zoo kab luam yeeb los lam xob.
Tsis muaj zoo rooj los lam zaum, tsis muaj zoo kab luam yeeb los lam haus.
Tsis noj los lam tuav diav, tsis luag los lam ntxi hniav.
Tsis noj mov ces yuav tshaib plab, tsis yuav kwv yuav tij ces yuav poob dab.
Tsis saib lub dej los saib lub lwg, tsis saib tus meej los saib tus tswg.
Tsis yog luag nom ces luag yeej tsis hawm, tsis yog luag kwv luag tij ces luag yeej tsis cawm.
Tsis yog luag nom luag yeej tsis xav hawm; tsis yog luag kwv luag tij luag yeej tsis xav cawm. 
Tso nyuj kom txog hav zaub, tso txhuv kom txog lauj kaub.
Tsov dig muag xuas tau ntiv qhiav, liab dig muag xuas tau txiv siav.
Tsov txaij nraum daim tawv, neeg txaij hauv lub plawv.
Tu qaib ntsuag tau ncej puab, tu tub ntsuag tau yeeb ncuab.
Tub nkeeg ntsib kev pluag, vaub tsuab ntsib kev mob tuag.
Tub nkeeg ntsib kev pluag; vuab tsuab ntsib kev mob tuag. 
Tub nquag ua khiav khuav, tub nkeeg tsuam lov duav.
Tub ntsoj tau zoo poj; tub ntsuag tau zoo hluas. 
Tub ntsoj tsis poj; tub ntsuag tsis tuag. 
Tub plaug tau tsiaj tuag; tub tshiab tau qav nchuav. 
Tub pluag tu tsiaj tuag, tub tshaib tau qav nchuav.
Tub qaug neej ces yuav dej yuav cawv, tub qaug dab ces yuav xyab yuav ntawv.
Tub sab ncauj ncab, tub nkeeg ncauj nreeg.
Tus nkees qhia npaum twg los tsis nquag, tus nquag ua npaum twg los tsis tuag.
Tus nquag ces rau siab ntso, tus nkees ces muaj quav tso.
Tus nquag paub tsis tseg, tus nkees qhia rau los tsis leg.
Tus nquag paub tsis txhua, tus nkees qhia rau los tsis ua.
Tus nyiam tsis tas hu, tus tsis nyiam caj dab tu.
Tus pluag coj tub luam kab, tus qhua coj tswv tsab.
Tus pluag coj tub luam kev, tus qhua coj tswv tsev.
Tus yeem tsis tas hais, tus tsis yeem cab pob ntseg ntais.
Txais yam twg yuav txais kom tau ko, hais lo twg yuav hais kom to.
Txawj noj ceg hlob, tsis txawj noj ces ua mob.
Txawj noj ceg tsau plab, tsis txawj no ces mob cab.
Txawj qeeg taw lo tshauv, zoo nkauj tau phem vauv.
Txawm ciaj yuav txawj tuag, txawj muaj yuav txawj tuag.
Txhawj mob yog txhawj tsam tuag, txhawj lub neej yog txhawj tsam tau txoj kev pluag.
Txhob khav tias tso quav yuav tsis thawj sub tsag,
Txhob noj luag nkaub tuav luag npe, tsam luag nyob nraum kwj tse.
Txhob thuam neeg loj leeb, txhob cem neeg quav yeeb, txhob cem neeg thov khawv, txhob thuam neeg ceg tawv, yuav cem kom cem dab, yuav thuam kom thuam neeg tub sab.
Txhob ua nplaum xyaw txua, txhob muab ntsiab xyaw xua.
Txhob vam zoo rau yav tom ntej, es muab txoj kev tub nkeeg los sej.
Txob vam zoo rau yav laus, es muab txoj kev tub nkeeg los taug.
Txog toj cia toj nphau, txog nom cia nom hau.
Txog toj cia toj pob, txog nom cia nom txhob.
Txoj cai luaj txoj plaub hau, txhua leej txhua tus hla tsis dhau.
Txuag zam zam tshiab, txuag siav siav ntev.
Txum muaj txum khwv ntxiv, txum pluag txum tsis txuag siv.
Ua lag ua luam yuav saib hmoov, tsham tua nqaij yuav saib hav zoov.
Ua liaj tau taug kev dav, ua teb taug kev mus saum cav.
Ua neeg nyob leej twg ua tub sab, luag pom ces luag ntshais nws cuag ntshai dab.
Ua neej nyob nqag nto ntuj, ploog nto ntsis, nyob txhiab txog ib txhis.
Ua neej nyob ua lub siab ncaj dhia dhau luag lub nkuaj, txawm tias noj qab nyob zoo los dhia tsis dhau tshuaj.
Ua neej nyob xav muaj tsis xav pluag, xav muaj txoj sia nyob tsis xav tuag.
Ua neej nyob yaj los kib txoob, cheb plag tsev qawj qab yiag to nyob muaj koob.
Ua niam tsev yuav ua qhov zoo, tsev neeg thiaj nto moo.
Ua nplej yeej muaj npluag, ua neej nyob yeej khiav tsis dhau txoj kev tuag.
Ua nyab thiaj pom kab, ua sev thiaj pom kev.
Ua qoob yeej muaj nroj, ua neej nyob yeej khiav tsis dhau txoj kev ploj.
Ua quav nyuj puab phawv, tshav ntuj tuaj ces quav nyuj nti zawv.
Ua ruam thiaj tau nyob zov luag tsev, ua ntse ces tau ua luag qhev.
Ua tau tsev thiaj pom zoo ntoo, muaj poj niam lawm mam pom hluas nkauj zoo.
Ua tawv, tawv ces tsis khov, yuav ua muag muag thiaj tsis lov.
Ua tsev txhaum laj nyob txog hnub puas, yuav poj niam txhaum xav txog yav hluas.
Ua txiv tsev yuav muaj tswv yim, yav tom ntej thiaj pab tau tsev neeg dim.
Ua yam twg yuav siv tswv yim, thaum ua tau thiaj tsis khuv sim.
Ua yam twg yuav xwb sab laj, thaum ua tiav lub siab thiab kaj.
Vam ntws li xub muv, nroo ntwg li xub ntab, noj sib hu ua sib pab.
Vaub tsuab luag cem piv rau poj ntxoog, tub nkeeg luag cem piv rau pob ntoos.
Vov txhob vov vuas zeb, xov txhob xov ntsa hlau, tsam muaj hnub dhia tsis dhau.
Xa nqaij rau hma, xa dej rau nram kwj ha.
Xa nqaij rau tsov, xa dej rau zaj qhov.
Xav tau qhov zoo los tau qhov phem, xav taug kev ncaj los tseem tau txoj kev lem.
Xav taug qhov zoo los kawg tau phem; xav taug txoj kev ncaj los kawg tau qhov kev nkhaus lem. 
Xav xav plag ntshav.
Xyeej npluag tsis xyeej noob, xyeej tsawg tsis xyeej coob.
Xyoob loj tsis cuag ntoo, muaj zog tsis cuag muaj hmoo.
Yog yus txhiaj tsis se, yog tub tsis tuag.
Yuav nkauj mog los ua poj niam, yam nkaus li muab kav tsawb los ua ko riam.
Yuav tub hluas ua hau, yam nkaus li muab quav kws los ua ko hlau.
Yug tub yug kiv los npaj laug, npaj qoob npaj loo tseg thiaj tau noj tau haus.
Zais nyiaj txiag txhob zais nruab tiag, zais nyiaj kub txhob zais ntawm ntus cub.
Zaj tuag los vim zaj lub pov, raub ris tuag los vim lub qhov.
Zom zeb mob tes; tuav cos mob taws; ris dej mob nruab quam; noj mov nkaus diav maum.
""".strip()

def check_syllables(lines):
    bad = False
    for line in lines:
        tokens = line.strip().split(' ')
        for token in tokens:
            s = token.translate(str.maketrans('', '', string.punctuation))
            s = s.lower()
            if s not in syllables:
                print(f'bad syllable {s}')
                print(line)
                print(tokens)
                bad = True
                break
        if bad:
            break

lines = s.split('\n')
lines = sorted(lines)

for line in lines:
    tokens = line.strip().split(' ')
    arr = []
    for token in tokens:
        s = token.translate(str.maketrans('', '', string.punctuation))
        s = s.lower()
        arr.append(s)
    r = '+'.join(arr)
    r = f'<generate.html?q={r}>'
    r = f'* `{line} {r}`_'
    print(r)
