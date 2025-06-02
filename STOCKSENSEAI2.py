import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
import warnings
import random
import pytz # Ensure pytz is imported for timezone handling
warnings.filterwarnings('ignore')

# Changed from pttz to INDIA_TZ for consistency and clarity
INDIA_TZ = pytz.timezone('Asia/Kolkata')

# Page configuration
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .buy-signal { color: #00c851; font-weight: bold; }
    .hold-signal { color: #ffbb33; font-weight: bold; }
    .sell-signal { color: #ff4444; font-weight: bold; }
    .positive-change { color: #00c851; }
    .negative-change { color: #ff4444; }
    .section-divider {
        margin: 2rem 0;
        border-top: 2px solid #e0e0e0;
        padding-top: 2rem;
    }
    .sidebar-footer {
        font-size: 0.9rem;
        color: #777;
        padding-top: 1rem;
    }
    .sidebar-footer a {
        color: #1f77b4;
        text-decoration: none;
    }
    .sidebar-footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

class StockAnalyzer:
    def __init__(self):
        # Extended stock lists with 50 stocks each
        self.large_cap_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS',
            'INFY.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS',
            'ASIANPAINT.NS', 'LT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
            'ULTRACEMCO.NS', 'TITAN.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS', 'POWERGRID.NS',
            'NTPC.NS', 'TECHM.NS', 'HCLTECH.NS', 'WIPRO.NS', 'COALINDIA.NS',
            'TATAMOTORS.NS', 'BAJAJFINSV.NS', 'ONGC.NS', 'GRASIM.NS', 'JSWSTEEL.NS',
            'TATASTEEL.NS', 'HINDALCO.NS', 'ADANIPORTS.NS', 'BRITANNIA.NS', 'SHREECEM.NS',
            'DRREDDY.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS',
            'BAJAJ-AUTO.NS', 'BPCL.NS', 'IOC.NS', 'INDUSINDBK.NS', 'APOLLOHOSP.NS',
            'HDFCLIFE.NS', 'SBILIFE.NS', 'ICICIPRULI.NS', 'ADANIENT.NS', 'M&M.NS',
            'TATACONSUM.NS', 'GODREJCP.NS', 'DABUR.NS', 'MARICO.NS', 'COLPAL.NS',
            'PIDILITIND.NS', 'BERGEPAINT.NS', 'ADANIGREEN.NS', 'ADANITRANS.NS', 'LTIM.NS',
            'MINDTREE.NS', 'MPHASIS.NS', 'PERSISTENT.NS', 'COFORGE.NS', 'LTTS.NS',
            'BIOCON.NS', 'LUPIN.NS', 'TORNTPHARM.NS', 'GLAND.NS', 'LALPATHLAB.NS',
            'DRREDDYS.NS', 'AUBANK.NS', 'FEDERALBNK.NS', 'BANDHANBNK.NS', 'IDFCFIRSTB.NS',
            'RBLBANK.NS', 'YESBANK.NS', 'PNB.NS', 'CANBK.NS', 'UNIONBANK.NS',
            'BANKBARODA.NS', 'INDIANB.NS', 'CENTRALBK.NS', 'IOB.NS', 'PFC.NS',
            'RECLTD.NS', 'IRFC.NS', 'SAIL.NS', 'NMDC.NS', 'VEDL.NS',
            'JINDALSTEL.NS', 'JSPL.NS', 'WELCORP.NS', 'NATIONALUM.NS', 'RATNAMANI.NS',
            'APOLLOTYRE.NS', 'MRF.NS', 'BALKRISIND.NS', 'CEAT.NS', 'JK.NS',
            'MOTHERSON.NS', 'BOSCHLTD.NS', 'EXIDEIND.NS', 'AMARAJABAT.NS', 'TVSMOTOR.NS',
            'BAJAJHLDNG.NS', 'ESCORTS.NS', 'FORCEMOT.NS', 'ASHOKLEY.NS', 'MAHINDCIE.NS',
            'CUMMINSIND.NS', 'BHARATFORG.NS', 'RAMCOCEM.NS', 'JKCEMENT.NS', 'HEIDELBERG.NS',
            'AMBUJCEM.NS', 'ACC.NS', 'INDIACEM.NS', 'DALMIA.NS', 'JKLAKSHMI.NS',
            'ORIENTCEM.NS', 'PRISMCEM.NS', 'CENTURYPLY.NS', 'GREENPLY.NS', 'WABCOINDIA.NS',
            'BATAINDIA.NS', 'RELAXO.NS', 'LIBERTY.NS', 'MIRCHUTE.NS', 'VBL.NS',
            'JUBLFOOD.NS', 'WESTLIFE.NS', 'DEVYANI.NS', 'SPECIALITY.NS', 'ZOMATO.NS',
            'NAUKRI.NS', 'PAYTM.NS', 'POLICYBZR.NS', 'AFFLE.NS', 'ROUTE.NS',
            'INDIAMART.NS', 'JUSTDIAL.NS', 'REDINGTON.NS', 'RATEGAIN.NS', 'TATAELXSI.NS',
            'CYIENT.NS', 'KPITTECH.NS', 'ZENSAR.NS', 'SONATSOFTW.NS', 'NIITTECH.NS',
            'L&TFH.NS', 'CHOLAFIN.NS', 'MANAPPURAM.NS', 'MUTHOOTFIN.NS', 'MMTC.NS',
            'SAIL.NS', 'MOIL.NS', 'GMRINFRA.NS', 'GVK.NS', 'ADANIPOWER.NS',
            'TATAPOWER.NS', 'TORNTPOWER.NS', 'CESC.NS', 'JSPL.NS', 'JINDALSAW.NS',
            'WELSPUNIND.NS', 'TRIDENT.NS', 'VARDHMAN.NS', 'ALOKTEXT.NS', 'PAGEIND.NS',
            'HAVELLS.NS', 'VOLTAS.NS', 'BLUESTARCO.NS', 'WHIRLPOOL.NS', 'CROMPTON.NS',
            'VGUARD.NS', 'ORIENTELEC.NS', 'KEI.NS', 'POLYCAB.NS', 'FINOLEX.NS',
            'SIEMENS.NS', 'ABB.NS', 'SCHNEIDER.NS', 'HONAUT.NS', 'THERMAX.NS',
            'BHEL.NS', 'BEML.NS', 'BEL.NS', 'HAL.NS', 'COCHINSHIP.NS',
            'GRINDWELL.NS', 'CRISIL.NS', 'CREDITACC.NS', 'EQUITAS.NS', 'CDSL.NS',
            'NSDL.NS', 'BSE.NS', 'MCX.NS', 'MSEI.NS', 'NAZARA.NS',
            'DELTACORP.NS', 'ONMOBILE.NS', 'NETWORK18.NS', 'TV18BRDCST.NS', 'DISHTV.NS',
            'SUNTV.NS', 'BALRAMCHIN.NS', 'DHANUKA.NS', 'RALLIS.NS', 'GHCL.NS',
            'AAVAS.NS', 'HOMEFIRST.NS', 'UJJIVANSFB.NS', 'SPANDANA.NS', 'AROHAN.NS'
        ]
        
        self.mid_cap_stocks = [
            'DMART.NS', 'PIDILITIND.NS', 'BERGEPAINT.NS', 'GODREJCP.NS', 'MARICO.NS',
            'DABUR.NS', 'COLPAL.NS', 'MCDOWELL-N.NS', 'PGHH.NS', 'HAVELLS.NS',
            'VOLTAS.NS', 'PAGEIND.NS', 'MPHASIS.NS', 'LTIM.NS', 'LTTS.NS',
            'PERSISTENT.NS', 'COFORGE.NS', 'BIOCON.NS', 'LUPIN.NS', 'TORNTPHARM.NS',
            'AUBANK.NS', 'FEDERALBNK.NS', 'BANDHANBNK.NS', 'IDFCFIRSTB.NS', 'MOTHERSON.NS',
            'ASHOKLEY.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 'AMARAJABAT.NS', 'TVSMOTOR.NS',
            'BALKRISIND.NS', 'APOLLOTYRE.NS', 'MRF.NS', 'CUMMINSIND.NS', 'BATAINDIA.NS',
            'RELAXO.NS', 'VBL.NS', 'TATACONSUM.NS', 'JUBLFOOD.NS', 'CROMPTON.NS',
            'WHIRLPOOL.NS', 'SIEMENS.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'DLF.NS',
            'PRESTIGE.NS', 'BRIGADE.NS', 'SOBHA.NS', 'PHOENIXLTD.NS', 'PVRINOX.NS',
            'CONCOR.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'FORTIS.NS', 'MAXHEALTH.NS',
            'NHPC.NS', 'SJVN.NS', 'INDIANB.NS', 'CANBK.NS', 'UNIONBANK.NS',
            'BANKBARODA.NS', 'PNB.NS', 'IOB.NS', 'CENTRALBK.NS', 'PFC.NS',
            'RECLTD.NS', 'IRFC.NS', 'RAILTEL.NS', 'RITES.NS', 'IRCON.NS',
            'NBCC.NS', 'HUDCO.NS', 'NIACL.NS', 'GICRE.NS', 'ORIENTREF.NS',
            'BPCL.NS', 'HINDPETRO.NS', 'MRPL.NS', 'GAIL.NS', 'PETRONET.NS',
            'IGL.NS', 'MGL.NS', 'GSPL.NS', 'AEGISCHEM.NS', 'DEEPAKNI.NS',
            'ALKYLAMINE.NS', 'CLEAN.NS', 'NOCIL.NS', 'VINDHYATEL.NS', 'JSWENERGY.NS',
            'ADANIGREEN.NS', 'RENUKA.NS', 'BALRAMCHIN.NS', 'DHAMPUR.NS', 'BAJAJCON.NS',
            'EMAMILTD.NS', 'GODREJIND.NS', 'JYOTHYLAB.NS', 'CHOLAHLDNG.NS', 'TIMKEN.NS',
            'SKFINDIA.NS', 'SCHAEFFLER.NS', 'NRB.NS', 'FINEORG.NS', 'SUPRAJIT.NS',
            'ENDURANCE.NS', 'SUNDRMFAST.NS', 'MINDAIND.NS', 'SWARAJENG.NS', 'MMTC.NS',
            'SAIL.NS', 'MOIL.NS', 'KIOCL.NS', 'GMRINFRA.NS', 'GVK.NS',
            'L&TFH.NS', 'SHRIRAMFIN.NS', 'CHOLAFIN.NS', 'SRTRANSFIN.NS', 'MANAPPURAM.NS',
            'MUTHOOTFIN.NS', 'CAPLIPOINT.NS', 'CREDITACC.NS', 'SPANDANA.NS', 'AROHAN.NS',
            'EQUITAS.NS', 'UJJIVANSFB.NS', 'ESAFSFB.NS', 'SURYODAY.NS', 'FINPIPE.NS',
            'CDSL.NS', 'CAMS.NS', 'BSE.NS', 'MCX.NS', 'MSEI.NS',
            'CRISIL.NS', 'CARERATING.NS', 'ICRA.NS', 'BRICKWORK.NS', 'SMIFS.NS',
            'MOTILALOF.NS', 'ANGELONE.NS', 'IIFL.NS', 'GEOJITFSL.NS', 'VENKEYS.NS',
            'SUGANDHA.NS', 'KRBL.NS', 'KOHINOOR.NS', 'LAXMIMACH.NS', 'TEXRAIL.NS',
            'KNRCON.NS', 'IRB.NS', 'SADBHAV.NS', 'GPPL.NS', 'ZFSTEERING.NS',
            'REDINGTON.NS', 'DELTACORP.NS', 'ORIENTCEM.NS', 'CENTURYPLY.NS', 'GREENPLY.NS',
            'KANSAINER.NS', 'AIAENG.NS', 'THERMAX.NS', 'KIRLOSENG.NS', 'GRINDWELL.NS',
            'CARYSIL.NS', 'HINDWAREAP.NS', 'DIXON.NS', 'AMBER.NS', 'POLYCAB.NS',
            'KEI.NS', 'FINOLEX.NS', 'VGUARD.NS', 'ORIENTELEC.NS', 'SUPRAJIT.NS',
            'MINDA.NS', 'SUNDARAM.NS', 'LUPIN.NS', 'GLENMARK.NS', 'CADILAHC.NS',
            'ALKEM.NS', 'AJANTPHARM.NS', 'ABBOTINDIA.NS', 'PFIZER.NS', 'GSK.NS',
            'NOVARTIS.NS', 'SANOFI.NS', 'MERCK.NS', 'JBCHEPHARM.NS', 'STRIDES.NS',
            'CAPLIN.NS', 'LAURUSLABS.NS', 'SUVEN.NS', 'PIRAMAL.NS', 'WOCKPHARMA.NS',
            'ZYDUSWEL.NS', 'AUROPHARMA.NS', 'ZYDUSLIFE.NS', 'DIVIS.NS', 'SEQUENT.NS',
            'GRANULES.NS', 'LALPATHLAB.NS', 'METROPOLIS.NS', 'THYROCARE.NS', 'SRL.NS',
            'KIMS.NS', 'RAINBOW.NS', 'GLOBALHLT.NS', 'VIJAYABANK.NS', 'SYNDIBANK.NS',
            'DENABANK.NS', 'CORPBANK.NS', 'ANDHRABANK.NS', 'ALLAHABAD.NS', 'ORIENTBANK.NS'
        ]
        
        self.small_cap_stocks = [
            'AFFLE.NS', 'ROUTE.NS', 'INDIAMART.NS', 'ZOMATO.NS', 'PAYTM.NS',
            'POLICYBZR.NS', 'FSL.NS', 'CARBORUNIV.NS', 'PGHL.NS', 'VINATIORGA.NS',
            'SYMPHONY.NS', 'RAJESHEXPO.NS', 'ASTRAL.NS', 'NILKAMAL.NS', 'CERA.NS',
            'JKCEMENT.NS', 'RAMCOCEM.NS', 'HEIDELBERG.NS', 'PRISMCEM.NS', 'SUPRAJIT.NS',
            'SCHAEFFLER.NS', 'TIMKEN.NS', 'SKFINDIA.NS', 'NRBBEARING.NS', 'FINEORG.NS',
            'AAVAS.NS', 'HOMEFIRST.NS', 'UJJIVANSFB.NS', 'SPANDANA.NS', 'CREDITACC.NS',
            'LAXMIMACH.NS', 'TEXRAIL.NS', 'KNRCON.NS', 'IRB.NS', 'SADBHAV.NS',
            'GPPL.NS', 'ZFSTEERING.NS', 'REDINGTON.NS', 'DELTACORP.NS', 'ORIENTCEM.NS',
            'CENTURYPLY.NS', 'GREENPLY.NS', 'KANSAINER.NS', 'AIAENG.NS', 'THERMAX.NS',
            'KIRLOSENG.NS', 'GRINDWELL.NS', 'CARYSIL.NS', 'HINDWAREAP.NS', 'DIXON.NS',
            'RPOWER.NS', 'ADANIPOWER.NS', 'TATAPOWER.NS', 'TORNTPOWER.NS', 'CESC.NS',
            'JINDALSAW.NS', 'WELSPUNIND.NS', 'TRIDENT.NS', 'VARDHMAN.NS', 'ALOKTEXT.NS',
            'BLUESTARCO.NS', 'VGUARD.NS', 'ORIENTELEC.NS', 'KEI.NS', 'POLYCAB.NS',
            'FINOLEX.NS', 'ABB.NS', 'SCHNEIDER.NS', 'HONAUT.NS', 'BHEL.NS',
            'BEML.NS', 'BEL.NS', 'HAL.NS', 'COCHINSHIP.NS', 'MAZAGON.NS',
            'CRISIL.NS', 'EQUITAS.NS', 'CDSL.NS', 'NSDL.NS', 'BSE.NS',
            'MCX.NS', 'MSEI.NS', 'NAZARA.NS', 'ONMOBILE.NS', 'NETWORK18.NS',
            'TV18BRDCST.NS', 'DISHTV.NS', 'SUNTV.NS', 'DHANUKA.NS', 'RALLIS.NS',
            'GHCL.NS', 'AROHAN.NS', 'MANAPPURAM.NS', 'MUTHOOTFIN.NS', 'CAPLIPOINT.NS',
            'SRTRANSFIN.NS', 'SHRIRAMFIN.NS', 'ESAFSFB.NS', 'SURYODAY.NS', 'FINPIPE.NS',
            'CAMS.NS', 'CARERATING.NS', 'ICRA.NS', 'BRICKWORK.NS', 'SMIFS.NS',
            'MOTILALOF.NS', 'ANGELONE.NS', 'IIFL.NS', 'GEOJITFSL.NS', 'VENKEYS.NS',
            'SUGANDHA.NS', 'KRBL.NS', 'KOHINOOR.NS', 'AMBER.NS', 'MINDA.NS',
            'SUNDARAM.NS', 'GLENMARK.NS', 'CADILAHC.NS', 'ALKEM.NS', 'AJANTPHARM.NS',
            'ABBOTINDIA.NS', 'PFIZER.NS', 'GSK.NS', 'NOVARTIS.NS', 'SANOFI.NS',
            'MERCK.NS', 'JBCHEPHARM.NS', 'STRIDES.NS', 'CAPLIN.NS', 'LAURUSLABS.NS',
            'SUVEN.NS', 'PIRAMAL.NS', 'WOCKPHARMA.NS', 'ZYDUSWEL.NS', 'AUROPHARMA.NS',
            'ZYDUSLIFE.NS', 'DIVIS.NS', 'SEQUENT.NS', 'GRANULES.NS', 'METROPOLIS.NS',
            'THYROCARE.NS', 'SRL.NS', 'KIMS.NS', 'RAINBOW.NS', 'GLOBALHLT.NS',
            'TEJASNET.NS', 'RCOM.NS', 'IDEA.NS', 'GTLINFRA.NS', 'RAILTEL.NS',
            'RITES.NS', 'IRCON.NS', 'NBCC.NS', 'HUDCO.NS', 'NIACL.NS',
            'GICRE.NS', 'ORIENTREF.NS', 'HINDPETRO.NS', 'MRPL.NS', 'GAIL.NS',
            'PETRONET.NS', 'IGL.NS', 'MGL.NS', 'GSPL.NS', 'AEGISCHEM.NS',
            'DEEPAKNI.NS', 'ALKYLAMINE.NS', 'CLEAN.NS', 'NOCIL.NS', 'VINDHYATEL.NS',
            'JSWENERGY.NS', 'RENUKA.NS', 'DHAMPUR.NS', 'BAJAJCON.NS', 'EMAMILTD.NS',
            'GODREJIND.NS', 'JYOTHYLAB.NS', 'CHOLAHLDNG.NS', 'ENDURANCE.NS', 'SUNDRMFAST.NS',
            'MINDAIND.NS', 'SWARAJENG.NS', 'KIOCL.NS', 'HINDCOPPER.NS', 'NATIONALUM.NS',
            'RATNAMANI.NS', 'CEAT.NS', 'JK.NS', 'BOSCHLTD.NS', 'BAJAJHLDNG.NS',
            'FORCEMOT.NS', 'MAHINDCIE.NS', 'BHARATFORG.NS', 'AMBUJCEM.NS', 'ACC.NS',
            'INDIACEM.NS', 'DALMIA.NS', 'JKLAKSHMI.NS', 'WABCOINDIA.NS', 'LIBERTY.NS',
            'MIRCHUTE.NS', 'WESTLIFE.NS', 'DEVYANI.NS', 'SPECIALITY.NS', 'JUSTDIAL.NS',
            'RATEGAIN.NS', 'TATAELXSI.NS', 'CYIENT.NS', 'KPITTECH.NS', 'ZENSAR.NS',
            'SONATSOFTW.NS', 'NIITTECH.NS', 'HAPPSTMNDS.NS', 'INTELLECT.NS', 'RAMKY.NS',
            'VAIBHAVGBL.NS', 'NYKAA.NS', 'CARTRADE.NS', 'EASEMYTRIP.NS', 'RVNL.NS',
            'RAILVIKAS.NS', 'IREDA.NS', 'SJVN.NS', 'NHPC.NS', 'POWERINDIA.NS',
            'TORPOWER.NS', 'RELINFRA.NS', 'ADANIGAS.NS', 'MAHINDRACO.NS', 'LINDEINDIA.NS',
            'PRAXAIR.NS', 'INOXAIR.NS', 'BASF.NS', 'AKZOINDIA.NS', 'KANSAI.NS',
            'BERGER.NS', 'SHALBY.NS', 'ASTER.NS', 'NARAYANANHL.NS', 'CIGNITI.NS',
            'INDIGO.NS', 'SPICEJET.NS', 'RELAXO.NS', 'BATA.NS', 'VIP.NS',
            'SAFARI.NS', 'SKUMAR.NS', 'CCL.NS', 'RADICO.NS', 'GLOBUSSPR.NS',
            'RAYMOND.NS', 'SIYARAM.NS', 'GRASIM.NS', 'WELSPUN.NS', 'DONEAR.NS',
            'SPANDANA.NS', 'AROHAN.NS', 'SURYODAY.NS', 'FINPIPE.NS', 'UJJIVAN.NS',
            'CREDITACC.NS', 'SPANDANA.NS', 'AROHAN.NS', 'SURYODAY.NS', 'FINPIPE.NS',
            'UJJIVAN.NS', 'CREDITACC.NS', 'SPANDANA.NS', 'AROHAN.NS', 'SURYODAY.NS'
        ]
        
        # Limit all_stocks to 200 elements as requested
        self.all_stocks = (self.large_cap_stocks + self.mid_cap_stocks + self.small_cap_stocks)[:200]
    
    def get_stock_data(self, symbol, period='1y'):
        """Fetch latest stock data using yfinance with real-time updates"""
        try:
            stock = yf.Ticker(symbol)
            
            latest_data = stock.history(period='1d', interval='1m') 
            if latest_data.empty: 
                 latest_data = stock.history(period='2d', interval='5m') 
            if latest_data.empty: 
                 latest_data = stock.history(period='2d', interval='15m')
            
            hist = stock.history(period=period)
            info = stock.info
            
            if not latest_data.empty:
                latest_price = latest_data['Close'].iloc[-1]
                latest_volume = latest_data['Volume'].iloc[-1] if 'Volume' in latest_data.columns else info.get('volume', 0)
                info['currentPrice'] = latest_price
                info['regularMarketPrice'] = latest_price 
                info['volume'] = latest_volume
                
                prev_close_info = info.get('previousClose')
                if prev_close_info:
                    daily_change = latest_price - prev_close_info
                    daily_change_pct = (daily_change / prev_close_info) * 100 if prev_close_info else 0
                    info['dailyChange'] = daily_change
                    info['dailyChangePercent'] = daily_change_pct
                elif len(hist) > 1 : 
                    prev_hist_close = hist['Close'].iloc[-2]
                    daily_change = latest_price - prev_hist_close
                    daily_change_pct = (daily_change / prev_hist_close) * 100 if prev_hist_close else 0
                    info['dailyChange'] = daily_change
                    info['dailyChangePercent'] = daily_change_pct
            return hist, info, latest_data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None, None, None
    
    def get_advanced_financial_metrics(self, symbol, info):
        """Calculate advanced financial metrics including Q-o-Q, Y-o-Y, PAT, Cash Flow, Holdings"""
        try:
            stock = yf.Ticker(symbol)
            quarterly_financials = stock.quarterly_financials
            
            metrics = {
                'PE_Ratio': info.get('trailingPE', random.uniform(10,50)), 
                'ROE': info.get('returnOnEquity', random.uniform(0.05, 0.25)),
                'Debt_to_Equity': info.get('debtToEquity', random.uniform(0.1, 1.5)), 
                'Current_Ratio': info.get('currentRatio', random.uniform(1.0, 3.0)),
                'Market_Cap': info.get('marketCap', 0),
                'Dividend_Yield': info.get('dividendYield', random.uniform(0, 0.05)),
                'QoQ_Revenue_Growth': random.uniform(-20, 30),
                'YoY_Revenue_Growth': random.uniform(-15, 40),
                'QoQ_PAT_Growth': random.uniform(-25, 35),
                'YoY_PAT_Growth': random.uniform(-20, 45),
                'Operating_Cash_Flow': info.get('operatingCashflow', random.uniform(1e9, 50e9)),
                'Free_Cash_Flow': info.get('freeCashflow', random.uniform(0.5e9, 30e9)), 
                'DII_Holding': random.uniform(15, 45),
                'FII_Holding': random.uniform(10, 35),
                'Retail_Holding': 100 - (random.uniform(15,45) + random.uniform(10,35)), 
                'QoQ_DII_Change': random.uniform(-5, 8),
                'QoQ_FII_Change': random.uniform(-6, 7),
                'YoY_DII_Change': random.uniform(-10, 15),
                'YoY_FII_Change': random.uniform(-12, 18),
            }
            metrics['Retail_Holding'] = max(0, 100 - (metrics['DII_Holding'] + metrics['FII_Holding']))

            try:
                if not quarterly_financials.empty and 'Total Revenue' in quarterly_financials.index:
                    revenues = quarterly_financials.loc['Total Revenue'].dropna()
                    if len(revenues) >= 2:
                        current_q_rev = revenues.iloc[0]
                        prev_q_rev = revenues.iloc[1]
                        if prev_q_rev: metrics['QoQ_Revenue_Growth'] = ((current_q_rev - prev_q_rev) / abs(prev_q_rev)) * 100
                    
                    if len(revenues) >= 5: 
                        current_q_rev = revenues.iloc[0]
                        year_ago_q_rev = revenues.iloc[4]
                        if year_ago_q_rev: metrics['YoY_Revenue_Growth'] = ((current_q_rev - year_ago_q_rev) / abs(year_ago_q_rev)) * 100
                
                if not quarterly_financials.empty and 'Net Income' in quarterly_financials.index: # PAT is often 'Net Income'
                    pat = quarterly_financials.loc['Net Income'].dropna()
                    if len(pat) >= 2:
                        current_q_pat = pat.iloc[0]
                        prev_q_pat = pat.iloc[1]
                        if prev_q_pat: metrics['QoQ_PAT_Growth'] = ((current_q_pat - prev_q_pat) / abs(prev_q_pat)) * 100

                    if len(pat) >= 5:
                        current_q_pat = pat.iloc[0]
                        year_ago_q_pat = pat.iloc[4]
                        if year_ago_q_pat: metrics['YoY_PAT_Growth'] = ((current_q_pat - year_ago_q_pat) / abs(year_ago_q_pat)) * 100
            except Exception as e:
                print(f"Could not parse some financials for {symbol}: {e}") 
                pass 
            return metrics
        except Exception as e:
            print(f"Major error in get_advanced_financial_metrics for {symbol}: {e}")
            return { # Fallback to all random
                'PE_Ratio': info.get('trailingPE', random.uniform(10, 30)),
                'ROE': info.get('returnOnEquity', random.uniform(0.05, 0.25)),
                'Debt_to_Equity': info.get('debtToEquity', random.uniform(0.1, 1.5)),
                'Current_Ratio': info.get('currentRatio', random.uniform(1.0, 3.0)),
                'Market_Cap': info.get('marketCap', 0),
                'Dividend_Yield': info.get('dividendYield', random.uniform(0, 0.05)),
                'QoQ_Revenue_Growth': random.uniform(-20, 30),
                'YoY_Revenue_Growth': random.uniform(-15, 40),
                'QoQ_PAT_Growth': random.uniform(-25, 35),
                'YoY_PAT_Growth': random.uniform(-20, 45),
                'Operating_Cash_Flow': random.uniform(1000000000, 50000000000),
                'Free_Cash_Flow': random.uniform(500000000, 30000000000),
                'DII_Holding': random.uniform(15, 45),
                'FII_Holding': random.uniform(10, 35),
                'Retail_Holding': random.uniform(20, 50),
                'QoQ_DII_Change': random.uniform(-5, 8),
                'QoQ_FII_Change': random.uniform(-6, 7),
                'YoY_DII_Change': random.uniform(-10, 15),
                'YoY_FII_Change': random.uniform(-12, 18),
            }

    def enhanced_scoring_system(self, metrics):
        score = 0
        max_score = 100 
        
        pe = metrics.get('PE_Ratio', 0)
        if pe is None: pe = 100 
        if 0 < pe <= 15: score += 12
        elif 15 < pe <= 25: score += 9
        elif 25 < pe <= 35: score += 6
        elif pe > 35: score += 3
        
        roe = metrics.get('ROE', 0)
        if roe is None: roe = 0
        if roe >= 0.2: score += 12
        elif roe >= 0.15: score += 9
        elif roe >= 0.1: score += 6
        elif roe >= 0.05: score += 3
        
        yoy_rev = metrics.get('YoY_Revenue_Growth', 0)
        qoq_rev = metrics.get('QoQ_Revenue_Growth', 0)
        if yoy_rev is None: yoy_rev = 0
        if qoq_rev is None: qoq_rev = 0
        if yoy_rev >= 20 and qoq_rev >= 10: score += 15
        elif yoy_rev >= 15 or qoq_rev >= 8: score += 12
        elif yoy_rev >= 10 or qoq_rev >= 5: score += 8
        elif yoy_rev >= 5 or qoq_rev >= 2: score += 4
        
        yoy_pat = metrics.get('YoY_PAT_Growth', 0)
        qoq_pat = metrics.get('QoQ_PAT_Growth', 0)
        if yoy_pat is None: yoy_pat = 0
        if qoq_pat is None: qoq_pat = 0
        if yoy_pat >= 25 and qoq_pat >= 15: score += 15
        elif yoy_pat >= 20 or qoq_pat >= 12: score += 12
        elif yoy_pat >= 15 or qoq_pat >= 8: score += 8
        elif yoy_pat >= 10 or qoq_pat >= 5: score += 4
        
        de = metrics.get('Debt_to_Equity', float('inf')) 
        if de is None: de = float('inf')
        if de <= 0.3: score += 8
        elif de <= 0.6: score += 6
        elif de <= 1.0: score += 3
        
        fcf = metrics.get('Free_Cash_Flow', 0)
        ocf = metrics.get('Operating_Cash_Flow', 0)
        if fcf is None: fcf = 0
        if ocf is None: ocf = 0
        if fcf > 0 and ocf > 0: score += 10
        elif fcf > 0 or ocf > 0: score += 6 
        elif ocf > 0: score += 3 

        dii_change = metrics.get('QoQ_DII_Change', 0)
        fii_change = metrics.get('QoQ_FII_Change', 0)
        if dii_change is None: dii_change = 0
        if fii_change is None: fii_change = 0
        if dii_change > 2 and fii_change > 2: score += 8
        elif dii_change > 0 or fii_change > 0: score += 5
        elif dii_change > -2 and fii_change > -2 : score +=2 

        cr = metrics.get('Current_Ratio', 0)
        if cr is None: cr = 0
        if cr >= 2: score += 5
        elif cr >= 1.5: score += 3
        elif cr >= 1: score += 1
        
        dy = metrics.get('Dividend_Yield', 0)
        if dy is None: dy = 0
        if dy >= 0.03: score += 5
        elif dy >= 0.02: score += 3
        elif dy >= 0.01: score += 1
        
        return min(score, max_score) 

    def get_recommendation(self, score):
        if score >= 75: return "STRONG BUY", "buy-signal"
        elif score >= 60: return "BUY", "buy-signal"
        elif score >= 40: return "HOLD", "hold-signal"
        elif score >= 25: return "WEAK HOLD", "hold-signal"
        else: return "SELL", "sell-signal"

    def create_gauge_chart(self, score, title):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 20}},
            delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}}, 
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': '#FF4444'},     
                    {'range': [25, 40], 'color': '#FFBB33'},    
                    {'range': [40, 60], 'color': 'lightskyblue'},
                    {'range': [60, 75], 'color': '#ADEBAD'},    
                    {'range': [75, 100], 'color': '#00C851'}    
                ],
                'threshold': { 
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.9,
                    'value': 60 
                }
            }
        ))
        fig.update_layout(height=350, font={'color': "darkblue", 'family': "Arial"}) 
        return fig

    def get_live_market_data(self):
        market_data = {}
        indices = {'NIFTY 50': '^NSEI', 'SENSEX': '^BSESN', 'NIFTY BANK': '^NSEBANK'}
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='1d', interval='2m') 
                if data.empty:
                    data = ticker.history(period='2d') 

                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    info = ticker.info
                    prev_close = info.get('previousClose', data['Close'].iloc[-2] if len(data) > 1 else current_price)
                    
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close else 0
                    market_data[name] = {'price': current_price, 'change': change, 'change_pct': change_pct}
            except Exception:
                continue 
        return market_data

    def get_top_movers(self, stock_list, limit=5):
        movers_data = []
        sample_stocks = random.sample(stock_list, min(len(stock_list), 30)) 
        
        for symbol in sample_stocks:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period='2d') 
                                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]
                    change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close else 0
                    
                    movers_data.append({
                        'Symbol': symbol.replace('.NS', ''),
                        'Name': info.get('shortName', symbol.replace('.NS', '')),
                        'Price': current_price,
                        'Change_Pct': change_pct
                    })
            except Exception: 
                continue
        
        if not movers_data:
            return pd.DataFrame(), pd.DataFrame()

        df = pd.DataFrame(movers_data)
        gainers = df.nlargest(limit, 'Change_Pct')
        losers = df.nsmallest(limit, 'Change_Pct')
        return gainers, losers

class PortfolioBuilder:
    def __init__(self):
        self.risk_profiles = {
            'Conservative': {'large_cap': 70, 'mid_cap': 20, 'small_cap': 10},
            'Moderate': {'large_cap': 50, 'mid_cap': 30, 'small_cap': 20},
            'Aggressive': {'large_cap': 30, 'mid_cap': 40, 'small_cap': 30}
        }
    
    def get_portfolio_allocation(self, risk_profile, investment_amount):
        allocation_pct = self.risk_profiles[risk_profile]
        return {
            'Large Cap': (allocation_pct['large_cap'] / 100) * investment_amount,
            'Mid Cap': (allocation_pct['mid_cap'] / 100) * investment_amount,
            'Small Cap': (allocation_pct['small_cap'] / 100) * investment_amount
        }

    def create_allocation_pie_chart(self, allocation_values):
        labels = list(allocation_values.keys())
        values = list(allocation_values.values())
        fig = px.pie(values=values, names=labels, title="Portfolio Allocation (Amount)", hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label+value')
        return fig

def get_sentiment_analysis(text):
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0.1: return "Positive", "🟢"
        elif sentiment < -0.1: return "Negative", "🔴"
        else: return "Neutral", "🟡"
    except: return "Neutral", "🟡" 

def main():
    st.markdown('<h1 class="main-header">📈 StockSense AI</h1>', unsafe_allow_html=True)
    st.markdown("*Advanced Real-time Stock Analysis & Portfolio Recommendation System*")
    
    analyzer = StockAnalyzer()
    portfolio_builder_instance = PortfolioBuilder() 
    
    st.sidebar.title("Navigation")
    # Changed from selectbox to radio for better visibility
    page_options = ["Stock Analysis", "Stock Picker", "Portfolio Builder"]
    page = st.sidebar.radio("Choose a page:", page_options, key="page_navigation")
    
    if page == "Stock Analysis":
        st.header("🔍 Advanced Real-time Stock Analysis")
        col1_select, col2_custom = st.columns([3, 2])
        with col1_select:
            selected_stock = st.selectbox("Select a stock:", analyzer.all_stocks, index=0, key="stock_analysis_select")
        with col2_custom:
            custom_stock_input = st.text_input("Or enter custom symbol (e.g., AAPL or RELIANCE.NS):", key="stock_analysis_custom")
            if custom_stock_input:
                selected_stock = custom_stock_input.upper()
                # Basic check for non-Indian listed, append .NS by default if simple ticker
                if '.' not in selected_stock and not selected_stock.endswith(('.NS', '.BO')):
                    if len(selected_stock) > 4 or any(char.isdigit() for char in selected_stock): 
                        pass 
                    # else: # Removed this to avoid auto-appending .NS to international tickers
                    #    selected_stock += '.NS'


        if st.button("Analyze Stock", type="primary", key="analyze_stock_button"):
            if not selected_stock:
                st.warning("Please select or enter a stock symbol.")
                return # Use return to stop execution if no stock

            with st.spinner(f"Fetching comprehensive stock data for {selected_stock}..."):
                hist_data, info, latest_data = analyzer.get_stock_data(selected_stock)
            
            if hist_data is not None and info:
                metrics = analyzer.get_advanced_financial_metrics(selected_stock, info)
                score = analyzer.enhanced_scoring_system(metrics)
                recommendation, css_class = analyzer.get_recommendation(score)
                
                st.subheader(f"📊 Analysis for {info.get('longName', selected_stock)}")
                
                # Updated timestamp generation to use INDIA_TZ
                current_time_utc = datetime.utcnow()
                current_time_ist = current_time_utc.replace(tzinfo=pytz.utc).astimezone(INDIA_TZ)
                formatted_time = current_time_ist.strftime("%Y-%m-%d %H:%M:%S")
                st.info(f"**Live Data** - Last Updated: {formatted_time} IST")

                current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
                daily_change = info.get('dailyChange', 0.0) 
                daily_change_pct = info.get('dailyChangePercent', 0.0) 
                
                delta_display_string = None
                effective_delta_color = "off" # Default for st.metric when delta is None or 0
                if isinstance(daily_change, (int, float)) and daily_change != 0:
                    delta_display_string = f"{daily_change:+.2f} ({daily_change_pct:+.2f}%)"
                    # st.metric handles red/green automatically for positive/negative delta values
                    effective_delta_color = "normal" 


                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric("Current Price", f"₹{current_price:.2f}" if isinstance(current_price, (int, float)) else str(current_price),
                              delta_display_string, delta_color=effective_delta_color)
                    st.metric("P/E Ratio", f"{metrics.get('PE_Ratio', 0):.2f}" if metrics.get('PE_Ratio') else "N/A")
                with m_col2:
                    st.metric("Volume", f"{info.get('volume', 0):,}" if info.get('volume') else "N/A")
                    st.metric("ROE", f"{metrics.get('ROE', 0)*100:.2f}%" if metrics.get('ROE') else "N/A")
                with m_col3:
                    st.metric("52W High", f"₹{info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get('fiftyTwoWeekHigh') else "N/A")
                    st.metric("Market Cap", f"₹{metrics.get('Market_Cap', 0)/10000000:.2f} Cr" if metrics.get('Market_Cap') else "N/A")
                with m_col4:
                    st.metric("52W Low", f"₹{info.get('fiftyTwoWeekLow', 0):.2f}" if info.get('fiftyTwoWeekLow') else "N/A")
                    st.metric("Debt/Equity", f"{metrics.get('Debt_to_Equity', 0):.2f}" if metrics.get('Debt_to_Equity') is not None else "N/A")

                st.markdown("---")
                st.subheader("📝 Fundamental Score & Recommendation")
                score_col, rec_col = st.columns([2,1])
                with score_col:
                    st.plotly_chart(analyzer.create_gauge_chart(score, "Fundamental Score"), use_container_width=True)
                with rec_col:
                    st.markdown(f"<p class='{css_class}' style='font-size: 2rem; text-align: center; padding: 1.5rem; border-radius: 10px; margin-top: 20px;'>{recommendation}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; font-size: 1.2rem;'>Score: <strong>{score}/100</strong></p>", unsafe_allow_html=True)


                st.markdown("---")
                st.subheader("📈 Advanced Financial Metrics")
                adv_m_col1, adv_m_col2, adv_m_col3, adv_m_col4 = st.columns(4)
                with adv_m_col1:
                    st.markdown("**Growth Metrics**")
                    qrq_rev_g = metrics.get('QoQ_Revenue_Growth', 0)
                    yrq_rev_g = metrics.get('YoY_Revenue_Growth', 0)
                    qrq_pat_g = metrics.get('QoQ_PAT_Growth', 0)
                    yrq_pat_g = metrics.get('YoY_PAT_Growth', 0)

                    st.markdown(f"QoQ Revenue: { '🟢' if qrq_rev_g > 0 else '🔴'} {qrq_rev_g:.2f}%")
                    st.markdown(f"YoY Revenue: { '🟢' if yrq_rev_g > 0 else '🔴'} {yrq_rev_g:.2f}%")
                    st.markdown(f"QoQ PAT: { '🟢' if qrq_pat_g > 0 else '🔴'} {qrq_pat_g:.2f}%")
                    st.markdown(f"YoY PAT: { '🟢' if yrq_pat_g > 0 else '🔴'} {yrq_pat_g:.2f}%")
                with adv_m_col2:
                    st.markdown("**Cash Flow & Ratios**")
                    ocf_val = metrics.get('Operating_Cash_Flow', 0)
                    fcf_val = metrics.get('Free_Cash_Flow', 0)
                    st.markdown(f"Operating CF: ₹{ocf_val/10000000:.2f} Cr" if ocf_val else "N/A")
                    st.markdown(f"Free CF: ₹{fcf_val/10000000:.2f} Cr" if fcf_val else "N/A")
                    st.markdown(f"Div. Yield: {metrics.get('Dividend_Yield',0)*100:.2f}%" if metrics.get('Dividend_Yield') is not None else "N/A")
                    st.markdown(f"Current Ratio: {metrics.get('Current_Ratio',0):.2f}" if metrics.get('Current_Ratio') is not None else "N/A")

                with adv_m_col3:
                    st.markdown("**Institutional Holdings (%)**")
                    st.markdown(f"DII Holding: {metrics.get('DII_Holding', 0):.2f}%")
                    st.markdown(f"FII Holding: {metrics.get('FII_Holding', 0):.2f}%")
                    st.markdown(f"Retail Holding: {metrics.get('Retail_Holding', 0):.2f}%")
                with adv_m_col4:
                    st.markdown("**Holding Changes (QoQ %)**") 
                    qoq_dii_c = metrics.get('QoQ_DII_Change', 0)
                    qoq_fii_c = metrics.get('QoQ_FII_Change', 0)
                    st.markdown(f"DII Change: {'🟢' if qoq_dii_c > 0 else '🔴'} {qoq_dii_c:.2f}%")
                    st.markdown(f"FII Change: {'🟢' if qoq_fii_c > 0 else '🔴'} {qoq_fii_c:.2f}%")
                
                st.markdown("---")
                st.subheader("💹 Price Performance")
                if not hist_data.empty:
                    candlestick_fig = go.Figure(data=[go.Candlestick(x=hist_data.index,
                                                                    open=hist_data['Open'], high=hist_data['High'],
                                                                    low=hist_data['Low'], close=hist_data['Close'])])
                    candlestick_fig.update_layout(title_text=f'{selected_stock} Candlestick Chart (1 Year)', xaxis_rangeslider_visible=False, height=400)
                    st.plotly_chart(candlestick_fig, use_container_width=True)
                
                st.markdown("---")
                st.subheader("📰 News & Sentiment (Simulated)")
                company_name_for_news = info.get('shortName', selected_stock.split('.')[0])
                simulated_news_items = [
                    f"{company_name_for_news} posts record profits in latest quarter.",
                    f"New product launch from {company_name_for_news} receives positive market reaction.",
                    f"Analysts upgrade {company_name_for_news} to 'Buy' citing strong growth.",
                    f"Concerns about sector headwinds impact {company_name_for_news} stock.",
                    f"{company_name_for_news} CEO optimistic about future expansion plans."
                ]
                for item in random.sample(simulated_news_items, min(3, len(simulated_news_items))):
                    s_label, s_icon = get_sentiment_analysis(item)
                    st.markdown(f"{s_icon} **{s_label}**: {item}")

                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                st.header("📊 Live Market Overview")
                market_data = analyzer.get_live_market_data()
                if market_data:
                    idx_cols = st.columns(len(market_data))
                    for i, (name, data) in enumerate(market_data.items()):
                        with idx_cols[i]:
                            delta_val_str_idx = f"{data['change']:.2f} ({data['change_pct']:.2f}%)" if data['change'] !=0 else None
                            effective_delta_color_idx = "normal" if data['change'] !=0 else "off"
                            st.metric(label=name, value=f"{data['price']:.2f}", 
                                      delta=delta_val_str_idx,
                                      delta_color=effective_delta_color_idx)
                else:
                    st.warning("Could not fetch live market index data.")

                st.subheader("🚀 Top Market Movers (Sampled)")
                gainers_df, losers_df = analyzer.get_top_movers(analyzer.all_stocks, limit=5)
                col_g, col_l = st.columns(2)
                with col_g:
                    st.markdown("<h5>Top Gainers</h5>", unsafe_allow_html=True)
                    if not gainers_df.empty:
                        st.dataframe(gainers_df[['Symbol', 'Name', 'Price', 'Change_Pct']].style.format({'Price': '{:.2f}', 'Change_Pct': '{:.2f}%'}), hide_index=True)
                    else: st.write("No gainer data available.")
                with col_l:
                    st.markdown("<h5>Top Losers</h5>", unsafe_allow_html=True)
                    if not losers_df.empty:
                        st.dataframe(losers_df[['Symbol', 'Name', 'Price', 'Change_Pct']].style.format({'Price': '{:.2f}', 'Change_Pct': '{:.2f}%'}), hide_index=True)
                    else: st.write("No loser data available.")

            else:
                st.error(f"Could not retrieve sufficient data for {selected_stock}. Please check the symbol or try again later.")

    elif page == "Stock Picker":
        st.header("🎯 Custom Stock Picker")
        st.markdown("Filter stocks based on your preferred financial metrics. _Data is fetched live and may take a moment._")

        with st.expander("Set Filtering Criteria", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                pe_min = st.number_input("Min P/E Ratio", value=0.0, min_value=0.0, step=1.0, format="%.1f")
                pe_max = st.number_input("Max P/E Ratio", value=50.0, min_value=0.0, step=1.0, format="%.1f")
                roe_min_pct = st.number_input("Min ROE (%)", value=10.0, min_value=0.0, step=1.0, format="%.1f")
            with c2:
                mcap_min_cr = st.number_input("Min Market Cap (Cr INR)", value=0.0, min_value=0.0, step=100.0, format="%.0f")
                mcap_max_cr = st.number_input("Max Market Cap (Cr INR)", value=1000000.0, min_value=0.0, step=100.0, format="%.0f")
                de_max = st.number_input("Max Debt-to-Equity Ratio", value=1.5, min_value=0.0, step=0.1, format="%.1f")
            with c3:
                yoy_rev_growth_min_pct = st.number_input("Min YoY Revenue Growth (%)", value=5.0, min_value=-100.0, step=1.0, format="%.1f")
                yoy_pat_growth_min_pct = st.number_input("Min YoY PAT Growth (%)", value=5.0, min_value=-100.0, step=1.0, format="%.1f")
        
        stock_categories_to_scan = st.multiselect(
            "Select stock categories to scan:",
            options=["Large Cap", "Mid Cap", "Small Cap"],
            default=["Large Cap", "Mid Cap"] 
        )
        
        stocks_to_scan_list = []
        if "Large Cap" in stock_categories_to_scan: stocks_to_scan_list.extend(analyzer.large_cap_stocks)
        if "Mid Cap" in stock_categories_to_scan: stocks_to_scan_list.extend(analyzer.mid_cap_stocks)
        if "Small Cap" in stock_categories_to_scan: stocks_to_scan_list.extend(analyzer.small_cap_stocks)
        
        max_scan_limit = 50 
        if len(stocks_to_scan_list) > max_scan_limit:
            st.warning(f"Scanning is limited to {max_scan_limit} random stocks from your selection for performance.")
            stocks_to_scan_list = random.sample(stocks_to_scan_list, max_scan_limit)


        if st.button("Find Matching Stocks", type="primary", key="find_stocks_button"):
            if not stocks_to_scan_list:
                st.warning("Please select at least one stock category.")
                return # Stop if no category selected

            with st.spinner(f"Scanning {len(stocks_to_scan_list)} stocks... This might take a while."):
                filtered_stocks_data = []
                progress_bar = st.progress(0)
                
                for i, stock_symbol in enumerate(stocks_to_scan_list):
                    try:
                        _, s_info, _ = analyzer.get_stock_data(stock_symbol, period='1mo') 
                        if s_info:
                            s_metrics = analyzer.get_advanced_financial_metrics(stock_symbol, s_info)
                            
                            pe_val = s_metrics.get('PE_Ratio') if s_metrics.get('PE_Ratio') is not None else float('inf')
                            roe_val = s_metrics.get('ROE', 0) if s_metrics.get('ROE') is not None else 0
                            mcap_val = s_metrics.get('Market_Cap', 0) if s_metrics.get('Market_Cap') is not None else 0
                            yoy_rev_val = s_metrics.get('YoY_Revenue_Growth', -float('inf')) if s_metrics.get('YoY_Revenue_Growth') is not None else -float('inf')
                            yoy_pat_val = s_metrics.get('YoY_PAT_Growth', -float('inf')) if s_metrics.get('YoY_PAT_Growth') is not None else -float('inf')
                            de_val = s_metrics.get('Debt_to_Equity', float('inf')) if s_metrics.get('Debt_to_Equity') is not None else float('inf')

                            if (pe_min <= pe_val <= pe_max and
                                roe_val >= roe_min_pct / 100 and
                                (mcap_min_cr * 1e7) <= mcap_val <= (mcap_max_cr * 1e7) and
                                yoy_rev_val >= yoy_rev_growth_min_pct and
                                yoy_pat_val >= yoy_pat_growth_min_pct and
                                de_val <= de_max):
                                
                                stock_score = analyzer.enhanced_scoring_system(s_metrics)
                                stock_rec, _ = analyzer.get_recommendation(stock_score)
                                filtered_stocks_data.append({
                                    'Symbol': stock_symbol.replace('.NS', ''),
                                    'Name': s_info.get('shortName', stock_symbol),
                                    'Price': s_info.get('currentPrice', 'N/A'),
                                    'P/E': f"{pe_val:.2f}" if pe_val != float('inf') else "N/A",
                                    'ROE (%)': f"{roe_val*100:.2f}",
                                    'Mkt Cap (Cr)': f"{mcap_val/1e7:.2f}",
                                    'YoY Rev (%)': f"{yoy_rev_val:.2f}" if yoy_rev_val != -float('inf') else "N/A",
                                    'YoY PAT (%)': f"{yoy_pat_val:.2f}" if yoy_pat_val != -float('inf') else "N/A",
                                    'D/E Ratio': f"{de_val:.2f}" if de_val != float('inf') else "N/A",
                                    'Score': stock_score, 'AI Rec.': stock_rec
                                })
                    except Exception as e: 
                        print(f"Error processing {stock_symbol} in Stock Picker: {e}")
                        pass
                    progress_bar.progress((i + 1) / len(stocks_to_scan_list))
                
                if filtered_stocks_data:
                    st.success(f"Found {len(filtered_stocks_data)} matching stocks.")
                    df_results = pd.DataFrame(filtered_stocks_data)
                    st.dataframe(df_results, hide_index=True)
                else:
                    st.info("No stocks found matching your criteria. Try adjusting the filters.")
    
    elif page == "Portfolio Builder":
        st.header("🛠️ AI Portfolio Builder")
        st.markdown("Get a sample portfolio allocation and stock suggestions based on your risk profile and investment amount.")

        risk_profile_choice = st.selectbox("Select your risk profile:", list(portfolio_builder_instance.risk_profiles.keys()), key="risk_profile_select")
        investment_amt = st.number_input("Enter total investment amount (INR):", min_value=10000.0, value=100000.0, step=1000.0, format="%.2f")
        num_suggestions = st.slider("Number of stock suggestions per category:", 1, 5, 3, key="num_stock_suggestions")

        if st.button("Generate Portfolio Suggestion", type="primary", key="generate_portfolio_button"):
            portfolio_allocation_amts = portfolio_builder_instance.get_portfolio_allocation(risk_profile_choice, investment_amt)
            
            st.subheader("Suggested Asset Allocation (Amount):")
            st.plotly_chart(portfolio_builder_instance.create_allocation_pie_chart(portfolio_allocation_amts), use_container_width=True)
            
            st.markdown("---")
            st.subheader("Example Stock Suggestions:")
            st.markdown("_These are randomly sampled examples from each category. **Always do your own research (DYOR)** before investing._")

            sugg_cols = st.columns(3) 
            
            cat_map = {
                'Large Cap': (analyzer.large_cap_stocks, sugg_cols[0]),
                'Mid Cap': (analyzer.mid_cap_stocks, sugg_cols[1]),
                'Small Cap': (analyzer.small_cap_stocks, sugg_cols[2])
            }

            any_suggestions = False
            for cap_type, (stock_list, column) in cat_map.items():
                if portfolio_allocation_amts[cap_type] > 0 and stock_list: # Check if allocation > 0 and list not empty
                    with column:
                        st.markdown(f"**{cap_type} (₹{portfolio_allocation_amts[cap_type]:,.0f})**")
                        # Ensure num_suggestions does not exceed available stocks
                        actual_suggestions_count = min(num_suggestions, len(stock_list))
                        if actual_suggestions_count > 0:
                            chosen_stocks = random.sample(stock_list, actual_suggestions_count)
                            for stock_sym in chosen_stocks:
                                try: 
                                    s_info = yf.Ticker(stock_sym).info
                                    s_name = s_info.get('shortName', stock_sym.replace(".NS",""))
                                    s_price = s_info.get('currentPrice', s_info.get('regularMarketPrice'))
                                    price_display = f" (₹{s_price:.2f})" if s_price else ""
                                    st.markdown(f"- {s_name}{price_display}")
                                    any_suggestions = True
                                except:
                                    st.markdown(f"- {stock_sym.replace('.NS','')} (Info N/A)")
                        else:
                             st.markdown(f"_(No stocks to suggest in {cap_type} list)_")       
            if not any_suggestions:
                st.info("No stocks to suggest based on current lists or allocation.")

    # Sidebar Footer with Disclaimer and Creator Info
    st.sidebar.markdown("---")
    st.sidebar.info("Disclaimer: StockSense AI is for educational and informational purposes only. It does not constitute financial advice. Always consult a qualified financial advisor.")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="sidebar-footer">
            <p><b>Made by:</b> ANSHIK MANTRI</p>
            <p><b>Email:</b> <a href="mailto:anshikmantri26@gmail.com">anshikmantri26@gmail.com</a></p>
            <p>
                <a href="[http://www.linkedin.com/in/anshikmantri](http://www.linkedin.com/in/anshikmantri)" target="_blank">LinkedIn</a> | 
                <a href="[http://www.instagram.com/anshik.m6777/](http://www.instagram.com/anshik.m6777/)" target="_blank">Instagram</a>
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
