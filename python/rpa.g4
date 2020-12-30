grammar rpa;

fragment C : ('c'|'d'|'f'|'h'|'k'|'l'|'m'|'n'|'p'|'q'|'r'|'s'|'t'|'v'|'x'|'y'|'z'|'ch'|'dh'|'hl'|'hm'|'hn'|'kh'|'ml'|'nc'|'nk'|'np'|'nq'|'nr'|'nt'|'ny'|'ph'|'pl'|'qh'|'rh'|'th'|'ts'|'tx'|'xy'|'hml'|'hny'|'nch'|'nkh'|'nph'|'npl'|'nqh'|'nrh'|'nth'|'nts'|'ntx'|'plh'|'tsh'|'txh'|'nplh'|'ntsh'|'ntxh');
fragment V : ('a'|'e'|'i'|'o'|'u'|'w'|'ai'|'au'|'aw'|'ee'|'ia'|'oo'|'ua');
fragment T : ('j'|'s'|'v'|'m'|'g'|'b');

WORD : (C V T | C V | V T) ;
WHITESPACE          : (' ' | '\t') ;
NEWLINE             : ('\r'? '\n' | '\r')+ ;
TEXT                : ~[\])]+ ;
