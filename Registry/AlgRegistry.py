from Algorithms.TD import TD
from Algorithms.GTD import GTD
from Algorithms.TDRC import TDRC
from Algorithms.GTD2 import GTD2
from Algorithms.PGTD2 import PGTD2
from Algorithms.HTD import HTD
from Algorithms.ETDLB import ETDLB
from Algorithms.ETD import ETD
from Algorithms.ETDH import ETDH
from Algorithms.ABTD import ABTD
from Algorithms.Vtrace import Vtrace
from Algorithms.TB import TB
from Algorithms.LSTD import LSTD
from Algorithms.LSETD import LSETD
from Algorithms.SARSA import SARSA
from Algorithms.ESARSA import ESARSA
from Algorithms.ESARSAH import ESARSAH
from Algorithms.ESARSA_VI import ESARSA_VI
from Algorithms.ESARSAH_VI import ESARSAH_VI
from Algorithms.Q import Q
from Algorithms.EQ import EQ
from Algorithms.EQH import EQH
from Algorithms.EQ_VI import EQ_VI
from Algorithms.EQH_VI import EQH_VI
from Algorithms.DQN_Agent import DQN_Agent
alg_dict = {'TD': TD, 'ETD': ETD, 'ETDH': ETDH, 'Vtrace': Vtrace, 'ABTD': ABTD, 'GTD': GTD, 'TB': TB, 'GTD2': GTD2, 'HTD': HTD,
            'ETDLB': ETDLB, 'PGTD2': PGTD2, 'TDRC': TDRC, 'LSTD': LSTD, 'LSETD': LSETD, 'SARSA': SARSA, 'ESARSA': ESARSA, 'ESARSAH': ESARSAH, 'ESARSA_VI': ESARSA_VI, 'ESARSAH_VI': ESARSAH_VI,
            'Q': Q, 'EQ': EQ, 'EQH': EQH, 'EQ_VI': EQ_VI, 'EQH_VI': EQH_VI, 'DQN_Agent': DQN_Agent}
