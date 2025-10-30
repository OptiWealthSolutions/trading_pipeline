fred_api_key="e16626c91fa2b1af27704a783939bf72"
polygon_api_key = "Qb5kB9YukAlm3Z_KTYVkiyApew2T378G"

#structure and temrinal's logic

# params

start_date = "2020-01-01"

main_pairs = ["EURUSD=X", "USDJPY=X", "EURGBP=X", "USDCAD=X"]

pairs = ["EURUSD=X", "USDJPY=X", "EURGBP=X", "USDCAD=X", "NZDUSD=X","AUDUSD=X", 'EURAUD=X', "EURNZD=X", "EURCHF=X" ]

indices = {
    "DXY (Dollar Index)": "DTWEXBGS",      # Broad USD Index
    "EUR Index": "DEXUSEU",                 # USD/EUR (inversé pour index EUR)
    "GBP Index": "DEXUSUK",                 # USD/GBP (inversé pour index GBP)
    "JPY Index": "DEXJPUS",                 # JPY/USD
    "CHF Index": "DEXSZUS",                 # CHF/USD
    "CAD Index": "DEXCAUS",                 # CAD/USD
}

# Macro data frame for the principal monetary area (japan, usa, euro, oceania, china, canada)

macro_data_config = {
    'USD': {
        'CPI': 'CPIAUCSL',
        'GDP': 'GDP',
        'PPI': 'PPIACO',
        'Interest Rate': 'FEDFUNDS',
        'Jobless Rate': 'UNRATE',
        'GDP Growth': 'A191RL1Q225SBEA'
    },
    'EUR': {
        'CPI': 'CP0000EZ19M086NEST',
        'GDP': 'CLVMNACSCAB1GQEA19',
        'PPI': 'PPI_EA19',
        'Interest Rate': 'ECBDFR',
        'Jobless Rate': 'LRHUTTTTEZM156S',
        'GDP Growth': 'NAEXKP01EZQ657S'
    },
    'GBP': {
        'CPI': 'CPALTT01GBM657N',
        'GDP': 'NAEXKP01GBQ661S',
        'PPI': 'PPIACOGBM086NEST',
        'Interest Rate': 'BOEBASE',
        'Jobless Rate': 'LRHUTTTTGBM156S',
        'GDP Growth': 'NAEXKP01GBQ657S'
    },
    'JPY': {
        'CPI': 'CPALTT01JPM657N',
        'GDP': 'CLVMNACSCAB1GQJP',
        'PPI': 'PPIACOJPM086NEST',
        'Interest Rate': 'IRSTCB01JPM156N',
        'Jobless Rate': 'LRHUTTTTJPQ156S',
        'GDP Growth': 'NAEXKP01JPQ657S'
    },
    'CAD': {
        'CPI': 'CPALTT01CAM657N',
        'GDP': 'CLVMNACSCAB1GQCA',
        'PPI': 'PPIACOCAM086NEST',
        'Interest Rate': 'IRSTCB01CAM156N',
        'Jobless Rate': 'LRHUTTTTCAQ156S',
        'GDP Growth': 'NAEXKP01CAQ657S'
    }
}
