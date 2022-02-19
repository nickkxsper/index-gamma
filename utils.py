import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
pd.options.display.float_format = '{:,.4f}'.format
filename = 'spx_quotedata.csv'
class Gamma:
    def __init__(self,
                 filename,
                 strike_strides
        ):
        self.filename = filename
        self.strike_strides = strike_strides
        self.generate_gamma_df()

    # Black-Scholes European-Options Gamma
    @staticmethod
    def BSGamma(S, K, vol, T, r, q, optType, OI):
        if T == 0 or vol == 0:
            return 0

        dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        dm = dp - vol*np.sqrt(T)

        if optType == 'call':
            gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
            return OI * 100 * S * S * 0.01 * gamma
        else: # Gamma is same for calls and puts. This is just to cross-check
            gamma = K * np.exp(-r*T) * norm.pdf(dm) / (S * S * vol * np.sqrt(T))
            return OI * 100 * S * S * 0.01 * gamma

    @staticmethod
    def isThirdFriday(d):
        return d.weekday() == 4 and 15 <= d.day <= 21


    def generate_gamma_df(self):
        # This assumes the CBOE file format hasn't been edited, i.e. table beginds at line 4
        optionsFile = open(self.filename)
        optionsFileData = optionsFile.readlines()
        optionsFile.close()

        # Get SPX Spot
        spotLine = optionsFileData[1]
        self.spotPrice = float(spotLine.split('Last:')[1].split(',')[0])
        self.fromStrike = 0.8 * self.spotPrice
        self.toStrike = 1.2 * self.spotPrice

        # Get Today's Date
        dateLine = optionsFileData[2]
        self.todayDate = dateLine.split('Date: ')[1].split(',')
        self.monthDay = self.todayDate[0].split(' ')

        # Handling of US/EU date formats
        if len(self.monthDay) == 2:
            year = int(self.todayDate[1])
            month = self.monthDay[0]
            day = int(self.monthDay[1])
        else:
            year = int(self.monthDay[2])
            month = self.monthDay[1]
            day = int(self.monthDay[0])

        todayDate = datetime.strptime(month,'%B')
        self.todayDate = todayDate.replace(day=day, year=year)

        # Get SPX Options Data
        df = pd.read_csv(filename, sep=",", header=None, skiprows=4)
        df.columns = ['ExpirationDate','Calls','CallLastSale','CallNet','CallBid','CallAsk','CallVol',
                      'CallIV','CallDelta','CallGamma','CallOpenInt','StrikePrice','Puts','PutLastSale',
                      'PutNet','PutBid','PutAsk','PutVol','PutIV','PutDelta','PutGamma','PutOpenInt']

        df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%a %b %d %Y')
        df['ExpirationDate'] = df['ExpirationDate'] + timedelta(hours=16)
        df['StrikePrice'] = df['StrikePrice'].astype(float)
        df['CallIV'] = df['CallIV'].astype(float)
        df['PutIV'] = df['PutIV'].astype(float)
        df['CallGamma'] = df['CallGamma'].astype(float)
        df['PutGamma'] = df['PutGamma'].astype(float)
        df['CallOpenInt'] = df['CallOpenInt'].astype(float)
        df['PutOpenInt'] = df['PutOpenInt'].astype(float)


        # ---=== CALCULATE SPOT GAMMA ===---
        # Gamma Exposure = Unit Gamma * Open Interest * Contract Size * Spot Price
        # To further convert into 'per 1% move' quantity, multiply by 1% of spotPrice
        df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * self.spotPrice * self.spotPrice * 0.01
        df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * self.spotPrice * self.spotPrice * 0.01 * -1

        df['TotalGamma'] = (df.CallGEX + df.PutGEX) / 10**9

        self.gamma_df = df
        self.gamma_df_strike_agg = df.groupby(['StrikePrice']).sum()
        self.strikes = self.gamma_df_strike_agg.index.values

    # Chart 1: Absolute Gamma Exposure
    def plot_absolute_gamma(self):
        plt.grid()
        plt.bar(self.strikes, self.gamma_df_strike_agg['TotalGamma'].to_numpy(), width=6, linewidth=0.1, edgecolor='k', label="Gamma Exposure")
        plt.xlim([self.fromStrike, self.toStrike])
        chartTitle = "Total Gamma: $" + str("{:.2f}".format(self.gamma_df['TotalGamma'].sum())) + " Bn per 1% SPX Move"
        plt.title(chartTitle, fontweight="bold", fontsize=20)
        plt.xlabel('Strike', fontweight="bold")
        plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
        plt.axvline(x=self.spotPrice, color='r', lw=1, label="SPX Spot: " + str("{:,.0f}".format(self.spotPrice)))
        plt.legend()
        plt.show()

    # Chart 2: Absolute Gamma Exposure by Calls and Puts
    def plot_absolute_gamma_by_c_p(self):
        plt.grid()
        plt.bar(self.strikes, self.gamma_df_strike_agg['CallGEX'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Call Gamma")
        plt.bar(self.strikes, self.gamma_df_strike_agg['PutGEX'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Put Gamma")
        plt.xlim([self.fromStrike, self.toStrike])
        chartTitle = "Total Gamma: $" + str("{:.2f}".format(self.gamma_df['TotalGamma'].sum())) + " Bn per 1% SPX Move"
        plt.title(chartTitle, fontweight="bold", fontsize=20)
        plt.xlabel('Strike', fontweight="bold")
        plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
        plt.axvline(x=self.spotPrice, color='r', lw=1, label="SPX Spot:" + str("{:,.0f}".format(self.spotPrice)))
        plt.legend()
        plt.show()

    def calc_gamma_profile(self):
        self.levels = np.linspace(self.fromStrike, self.toStrike, self.strike_strides)

        # For 0DTE options, I'm setting DTE = 1 day, otherwise they get excluded
        self.gamma_df['daysTillExp'] = [1/252 if (np.busday_count(self.todayDate.date(), x.date())) == 0 \
                                   else np.busday_count(self.todayDate.date(), x.date())/252 for x in self.gamma_df.ExpirationDate]

        nextExpiry = self.gamma_df['ExpirationDate'].min()

        self.gamma_df['IsThirdFriday'] = [self.isThirdFriday(x) for x in self.gamma_df.ExpirationDate]
        thirdFridays = self.gamma_df.loc[self.gamma_df['IsThirdFriday'] == True]
        nextMonthlyExp = thirdFridays['ExpirationDate'].min()

        totalGamma = []
        totalGammaExNext = []
        totalGammaExFri = []

        # For each spot level, calc gamma exposure at that point
        for level in self.levels:
            self.gamma_df['callGammaEx'] = self.gamma_df.apply(lambda row : self.BSGamma(level, row['StrikePrice'], row['CallIV'],
                                                                   row['daysTillExp'], 0, 0, "call", row['CallOpenInt']), axis = 1)

            self.gamma_df['putGammaEx'] = self.gamma_df.apply(lambda row : self.BSGamma(level, row['StrikePrice'], row['PutIV'],
                                                                  row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), axis = 1)

            totalGamma.append(self.gamma_df['callGammaEx'].sum() - self.gamma_df['putGammaEx'].sum())

            exNxt = self.gamma_df.loc[self.gamma_df['ExpirationDate'] != nextExpiry]
            totalGammaExNext.append(exNxt['callGammaEx'].sum() - exNxt['putGammaEx'].sum())

            exFri = self.gamma_df.loc[self.gamma_df['ExpirationDate'] != nextMonthlyExp]
            totalGammaExFri.append(exFri['callGammaEx'].sum() - exFri['putGammaEx'].sum())

        self.totalGamma = np.array(totalGamma) / 10**9
        self.totalGammaExNext = np.array(totalGammaExNext) / 10**9
        self.totalGammaExFri = np.array(totalGammaExFri) / 10**9

        # Find Gamma Flip Point
        self.zeroCrossIdx = np.where(np.diff(np.sign(self.totalGamma)))[0][0]

        negGamma = totalGamma[self.zeroCrossIdx]
        posGamma = totalGamma[self.zeroCrossIdx+1]
        negStrike = self.levels[self.zeroCrossIdx]
        posStrike = self.levels[self.zeroCrossIdx+1]

        self.zeroGamma = posStrike - ((posStrike - negStrike) * posGamma/(posGamma-negGamma))
        self.zeroGamma = self.zeroGamma[0]


    def plot_gamma_profile(self):
        fig, ax = plt.subplots()
        plt.grid()
        plt.plot(self.levels, self.totalGamma, label="All Expiries")
        plt.plot(self.levels, self.totalGammaExNext, label="Ex-Next Expiry")
        plt.plot(self.levels, self.totalGammaExFri, label="Ex-Next Monthly Expiry")
        chartTitle = "Gamma Exposure Profile, SPX, " + self.todayDate.strftime('%d %b %Y')
        plt.title(chartTitle, fontweight="bold", fontsize=20)
        plt.xlabel('Index Price', fontweight="bold")
        plt.ylabel('Gamma Exposure ($ billions/1% move)', fontweight="bold")
        plt.axvline(x=self.spotPrice, color='r', lw=1, label="SPX Spot: " + str("{:,.0f}".format(self.spotPrice)))
        plt.axvline(x=self.zeroGamma, color='g', lw=1, label="Gamma Flip: " + str("{:,.0f}".format(self.zeroGamma)))
        plt.axhline(y=0, color='grey', lw=1)
        plt.xlim([self.fromStrike, self.toStrike])
        trans = ax.get_xaxis_transform()
        plt.fill_between([self.fromStrike, self.zeroGamma], min(self.totalGamma), max(self.totalGamma), facecolor='red', alpha=0.1, transform=trans)
        plt.fill_between([self.zeroGamma, self.toStrike], min(self.totalGamma), max(self.totalGamma), facecolor='green', alpha=0.1, transform=trans)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    gtool = Gamma(filename='spx_quotedata.csv'
                  strike_strides = 60)