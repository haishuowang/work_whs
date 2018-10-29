"""
Public Server File Path Proxy

"""

from pathlib import Path
from functools import reduce


class _BaseDirs:
    # Root Dir
    def __init__(self, mode=None):
        default_path = Path("/mnt/mfs")
        if mode == "bkt":
            self.BASE_PATH = Path("/mnt/mfs")
        elif mode == "pro":
            self.BASE_PATH = Path("/media/hdd1")
        else:
            self.BASE_PATH = default_path


class _StockDir:
    __DIRS_TO_REGISTER = {
        "EM_Funda": "EM_Funda",
        "EM_Tab01": "EM_Tab01",
        "EM_Tab03": "EM_Tab03",
        "EM_Tab04": "EM_Tab04",
        "EM_Tab06": "EM_Tab06",
        "EM_Tab07": "EM_Tab07",
        "EM_Tab09": "EM_Tab09",
        "EM_Tab10": "EM_Tab10",
        "EM_Tab11": "EM_Tab11",
        "EM_Tab12": "EM_Tab12",
        "EM_Tab14": "EM_Tab14",
    }

    # Data Dir
    def __init__(self, mode=None):
        self.BASE_PATH = _BaseDirs(mode).BASE_PATH / "DAT_EQT"
        for dir_name, dir_path in self.__DIRS_TO_REGISTER.items():
            self.__setattr__(dir_name, self.BASE_PATH / dir_path)


class _Directory:
    def __init__(self, root, children):
        self.__children = set()
        for child in children:
            val = root / child
            setattr(self, child, val)
            self.__children.add(val)

    def __contains__(self, item):
        return Path(item) in self.__children


class _BinFiles:
    """
    Binary Files stable

    e.g. BinFiles.`数据库表明`

    *** USE THIS IN YOUR PRODUCTION CODE ***

    """
    __EM_Tables = {
        "EM_Tab01": ["CDSY_CHANGEINFO", "CDSY_SECUCODE"],
        "EM_Tab03": ["LICO_CM_COINTRO", "LICO_CM_PRODUCT", "LICO_CM_INDUSTRY", "LICO_CM_PERFORMANCEF",
                     "LICO_CM_REGION"],
        "EM_Tab04": ["LICO_FN_RGINCOME", "LICO_FN_RGCASHFLOW", "LICO_FN_RGBALANCE", "LICO_FN_NGCASHFLOWADD",
                     "LICO_FN_PERFORMANCEE"],
        "EM_Tab06": ["LICO_ES_CPHPSHNUM", "LICO_ES_CPHSSTRUCT", "LICO_ES_STRUCTCI", "LICO_ES_SXJJSJB",
                     "LICO_FP_COMDECLARE", "LICO_IA_ASSIGNSCHEME", "LICO_FP_ISSUELOCK", "LICO_FP_ISSUEBASICINFO"],
        "EM_Tab07": ["LICO_ES_LTEINVEST", "LICO_CM_GUARANTYMATTER", "LICO_CM_GUARANTEE", "LICO_CM_GUARANTEEACCU"],
        "EM_Tab09": ["INDEX_TD_DAILY", "INDEX_TD_DAILYSYS", "IDEX_YS_WEIGHT_A", "INDEX_BA_SAMPLE"],
        "EM_Tab10": ["DERIVED_10", "LICO_YS_STOCKVALUE"],
        "EM_Tab11": ["LICO_FN_GENERALFINA", "LICO_FN_FCRGINCOMES", "LICO_FN_FCRGCASHS", "LICO_FN_SIGQUAFINA"],
        "EM_Tab12": ["DERIVED_12", "LICO_IM_INCHG"],
        "EM_Tab14": ["DERIVED_14", "TRAD_SK_DAILYCH", "TRAD_SK_FACTOR1", "TRAD_SK_DAILY_JC",
                     "TRAD_SK_RANK", "TRAD_TD_TDATE", "TRAD_SK_REVALUATION", "TRAD_TD_SUSPEND",
                     "TRAD_TD_SUSPENDDAY", "TRAD_MT_MARGIN", "TIT_S_PUB_SALES", "TIT_S_PUB_STOCK"],
        "EM_Funda": ["daily"]
    }

    __DIRS_TO_REGISTER = {**__EM_Tables,
                          "EM_Funda": reduce(lambda x, y: [*x, *y], __EM_Tables.values())}

    def __init__(self, mode):
        self.StockDir = _StockDir(mode)

        for dir_name, dir_path in self.__DIRS_TO_REGISTER.items():
            dir_ = _Directory(getattr(self.StockDir, dir_name), dir_path)
            self.__setattr__(dir_name, dir_)


class AutoPath:
    """
    Recursively register all children directory as attribute under given path

    e.g.
    p = AutoPath("/mnt/mfs/DAT_EQT")

    1) Resolve Child Directory as instance attribute
    p.EM_FUNDA
    p.EM_FUNDA.DAILY
    p.EM_Tab01

    2) Represent path in string
    str(p.EM_FUNDA.path) -> '/mnt/mfs/DAT_EQT/EM_Funda'

    3) Use as pathlib.Path(PosixPath/WindosPath, depends on the runtime platform)
    p.EM_FUNDA.path -> PosixPath('/mnt/mfs/DAT_EQT/EM_Funda')
    p.EM_FUNDA.DAILY.path -> PosixPath('/mnt/mfs/DAT_EQT/EM_Funda/daily')


    """

    def _register(self, path: Path):
        """
        Register instance attrubute by its children directories name in upper case


        e.g.
        A directory structure as below

        /mnt/mfs/
            DAT_Eqt
            DAT_Fut
            some_file.pkl
            some_file.csv
            ...

        than `obj` will register two attribute DAT_EQT, DAT_FUT,
        and can be called as: obj.DAT_EQT, obj.DAT_FUT


        Args:
            path: pathlib.Path

        """

        try:
            childred_dirs = filter(lambda x: x.is_dir(), path.iterdir())
            for dir_ in childred_dirs:
                print(dir_)
                setattr(self, dir_.name.upper(), AutoPath(dir_))

        except PermissionError:
            pass

    def __init__(self, path_dir):
        path_dir = Path(path_dir) if type(path_dir) is str else path_dir
        self.path = path_dir
        self._register(path_dir)

    def __repr__(self):
        return str(self.path)


_DOCS = {
    # 01
    "CDSY_CHANGEINFO": ("证券代码表", "基本信息"),

    # 14
    "TRAD_SK_DAILYCH": ("个股日行情筹码分布", "交易数据"),
    "TRAD_SK_DAILY_JC": ("股票日行情", "交易数据")
}
