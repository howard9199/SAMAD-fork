POS = {
    "": 'NO_TAG',
    "ADJ": 'ADJ',
    "ADP": 'ADP',
    "ADV": 'ADV',
    "AUX": 'AUX',
    "CONJ": 'CONJ',  # U20
    "CCONJ": 'CCONJ',
    "DET": 'DET',
    "INTJ": 'INTJ',
    "NOUN": 'NOUN',
    "NUM": 'NUM',
    "PART": 'PART',
    "PRON": 'PRON',
    "PROPN": 'PROPN',
    "PUNCT": 'PUNCT',
    "SCONJ": 'SCONJ',
    "SYM": 'SYM',
    "VERB": 'VERB',
    "X": 'X',
    "EOL": 'EOL',
    "SPACE": 'SPACE'
}
morph = ['VerbForm=Ger', 'VerbForm=Conv', 'Mood=Prp', 'Animacy=Hum', 'Case=Gen', 'Aspect=Prog', 'NumType=Ord', 'Number=Count', 'Case=Loc', 'Number=Pauc', 'Case=Abe', 'Case=Ill', 'Case=Voc', 'Case=Sup', 'Case=Par', 'NounClass=Bantu20', 'Case=Ine', 'NounClass=Bantu15', 'Voice=Lfoc', 'Degree=Dim', 'Mood=Cnd', 'NounClass=Bantu10', 'Case=Erg', 'Voice=Rcp', 'Number=Inv', 'PronType=Emp', 'Mood=Des', 'Tense=Past', 'Degree=Abs', 'Polite=Form', 'Clusivity=In', 'NounClass=Bantu3', 'Animacy=Nhum', 'Case=Tem', 'NounClass=Wol4', 'Person=1', 'Evident=Nfh', 'Mood=Opt', 'NounClass=Bantu13', 'NounClass=Wol10', 'Voice=Act', 'NounClass=Bantu4', 'Polarity=Pos', 'Polite=Humb', 'NumType=Frac', 'Gender=Com', 'Case=Equ', 'Case=Per', 'Case=Ela', 'Voice=Bfoc', 'Degree=Aug', 'Number=Coll', 'Number=Tri', 'Typo=Yes', 'Aspect=Iter', 'Case=Ins', 'Voice=Cau', 'NumType=Sets', 'PronType=Neg', 'Voice=Antip', 'Case=Cns', 'Degree=Pos', 'Aspect=Imp', 'Definite=Def', 'VerbForm=Fin', 'Case=Cmp', 'Voice=Pass', 'Mood=Pot', 'Case=Spl', 'Person=0', 'Person=4', 'Definite=Spec', 'Case=Add', 'NounClass=Bantu19', 'NumType=Dist', 'Number=Dual', 'PronType=Ind', 'Number=Sing', 'NounClass=Bantu6', 'Number=Grpa', 'VerbForm=Inf', 'Animacy=Anim', 'PronType=Prs', 'Case=Com', 'NounClass=Bantu5', 'PronType=Tot', 'NounClass=Wol8', 'Polite=Elev', 'NounClass=Wol2', 'NounClass=Bantu14', 'Case=Acc', 'Case=Sub', 'NounClass=Bantu16', 'NounClass=Wol7', 'VerbForm=Part', 'NumType=Range', 'Voice=Inv', 'NounClass=Bantu9', 'Polite=Infm', 'NounClass=Wol9', 'VerbForm=Gdv', 'Tense=Pres', 'Abbr=Yes', 'NumType=Mult', 'Definite=Com', 'Case=Ade', 'NounClass=Wol12', 'NounClass=Bantu18', 'Case=Dat', 'NounClass=Bantu23', 'Case=Sbl', 'Case=Ter', 'Gender=Fem', 'Case=Abl', 'Mood=Jus', 'Mood=Imp', 'NounClass=Bantu7', 'PronType=Int', 'NounClass=Wol3', 'NumType=Card', 'NounClass=Bantu17', 'Aspect=Perf', 'Mood=Ind', 'PronType=Rcp', 'Aspect=Hab', 'Degree=Cmp', 'Evident=Fh', 'Case=Nom', 'Tense=Fut', 'Case=Dis', 'Tense=Imp', 'Case=Ess', 'Mood=Int', 'Gender=Neut', 'NounClass=Bantu12', 'VerbForm=Vnoun', 'Gender=Masc', 'Case=All', 'Tense=Pqp', 'Mood=Qot', 'Number=Ptan', 'Voice=Mid', 'NounClass=Bantu2', 'Case=Sbe', 'Mood=Nec', 'Mood=Sub', 'Number=Plur', 'Foreign=Yes', 'Degree=Equ', 'Reflex=Yes', 'Number=Grpl', 'Voice=Dir', 'Aspect=Prosp', 'NounClass=Wol5', 'Person=3', 'Case=Cau', 'VerbForm=Sup', 'Poss=Yes', 'NounClass=Bantu8', 'PronType=Art', 'Case=Tra', 'Case=Abs', 'PronType=Rel', 'Mood=Adm', 'PronType=Exc', 'NounClass=Bantu22', 'PronType=Dem', 'Definite=Ind', 'NounClass=Wol6', 'NounClass=Bantu1', 'Person=2', 'Case=Lat', 'Case=Ben', 'Polarity=Neg', 'Clusivity=Ex', 'Definite=Cons', 'Case=Core: ', 'Case=Del', 'Animacy=Inan', 'Mood=Irr', 'Degree=Sup']
DEP = ['ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt', 'punct', 'quantmod', 'relcl', 'xcomp']

# print(len(DEP))
# print(len(morph))
# print(len(POS))