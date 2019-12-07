import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from sklearn import tree
from sklearn import preprocessing
import graphviz

party_control_org = pd.read_csv("C:\\Users\\shawn\\Desktop\\MIT college work\\17.801\\Final Project\\House, Senate, Executive Party control.csv",  encoding = "cp1252")


start = int(party_control_org["Year"][0][:4])
end = int(party_control_org["Year"][len(party_control_org["Year"])-1][:4])
party_control_list = []

for i in range(start,end):
    index = (i - start)//2
    pres = party_control_org["President"][index]
    senate = party_control_org["Senate"][index]
    house = party_control_org["House"][index]
    if pres == senate and pres == house:
        gov  = "Unified"
        gov_marker = 1
        if pres == "R":
            party_marker = 1
        else:
            party_marker = 0
    else:
        gov = "Divided"
        gov_marker = 0
        party_marker = None
    party_control_list.append([i,pres,senate, house, gov, gov_marker, party_marker])

party_control = pd.DataFrame(party_control_list,columns=["Year", "President", "Senate", "House", "Divided/Unified", "Gov Marker", "Party Marker"])

sp_returns = pd.read_csv("C:\\Users\\shawn\\Desktop\\MIT college work\\17.801\\Final Project\\S&P 500 Yearly Returns.csv")
df_congress = pd.read_csv("C:\\Users\\shawn\\Desktop\\MIT college work\\17.801\\Final Project\\congressional_approval.csv")
df_congress_year = df_congress[['Year', 'True Approval']].groupby(['Year']).mean()
df_debt = pd.read_csv("C:\\Users\\shawn\\Desktop\\MIT college work\\17.801\\Final Project\\us_debt.csv", encoding = "cp1252")
df_military = pd.read_csv("C:\\Users\\shawn\\Desktop\\MIT college work\\17.801\\Final Project\\military_size.csv")


df_data = pd.merge(party_control,sp_returns, how= 'outer', on = "Year")
df_data = pd.merge(df_data, df_congress_year, how = 'outer', on = "Year")
df_data = pd.merge(df_data, df_debt, how = 'outer', on = 'Year')
df_data = pd.merge(df_data, df_military, how = 'outer', on = 'Year')
df_data["Change in True Approval"] = df_data["True Approval"].diff()
df_data['Debt/GDP Ratio'] = df_data['Debt/GDP Ratio'].str.rstrip('%').astype('float') / 100.0
df_data["Debt"] = df_data["Debt"].replace('[\$,]', '', regex=True).astype(float)
df_data["Military Total"] = df_data["Military Total"].str.replace(',','').astype('float')
df_data["Change in Debt"], df_data["Change in Debt/GDP Ratio"], df_data["Change in Debt Percent"], df_data["Change in S&P Returns"], df_data["Change in Military Total"]= \
    df_data["Debt"].diff(), df_data["Debt/GDP Ratio"].diff(), df_data["Debt"].pct_change(), df_data["S&P Returns"].diff(), df_data["Military Total"].diff()
df_data['President'], df_data['House'], df_data['Senate'] = df_data['President']+'President', df_data['House']+'House', df_data['Senate']+'Senate'
def scatter(y, x, marker):
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=marker, cmap="Spectral")
    plt.show()

scatter(df_data["Change in Debt Percent"], df_data["Year"], df_data["Gov Marker"])

model_econ_parties = ols('Q("S&P Returns") ~ President+ Senate+ House', data = df_data, missing = 'drop')
model_delta_approval_parties = ols('Q("Change in True Approval") ~ C(President)+ C(Senate)+ C(House)', data = df_data, missing = 'drop')
model_approval_parties = ols('Q("True Approval") ~ C(President)+ C(Senate)+ C(House) + Year', data = df_data, missing = 'drop')
model_debt_parties = ols('Debt ~ C(President)+ C(Senate)+ C(House) +Year', data = df_data, missing = 'drop')
model_delta_debt_parties = ols('Q("Change in Debt") ~ C(President)+ C(Senate)+ C(House) +Year', data = df_data, missing = 'drop')
model_delta_percent_debt_parties = ols('Q("Change in Debt Percent") ~ C(President)+ C(Senate)+ C(House) +Year', data = df_data, missing = 'drop')
model_delta_debt_gdp_parties = ols('Q("Change in Debt/GDP Ratio") ~ President+ Senate+ House', data = df_data, missing = 'drop')

def lin_reg_categorical(model_name):
    fitted_model = model_name.fit()
    print(fitted_model.summary())
def lin_reg_categorical_input(y, x_inputs):
    ols_string = f'Q("{y}") ~ '
    for x in x_inputs:
        ols_string += f'Q("{x}") +'
    ols_string = ols_string[:-1]
    model = ols(ols_string, data = df_data, missing = 'drop')
    fitted_model = model.fit()
    print(fitted_model.summary())
lin_reg_categorical(model_delta_debt_parties) #decent
lin_reg_categorical(model_debt_parties) #really good
lin_reg_categorical(model_delta_percent_debt_parties)

lin_reg_categorical_input("True Approval", ['President', 'House','Senate', 'Year'])
lin_reg_categorical_input("Military Total", ['President', 'House','Senate']) #ok
lin_reg_categorical_input("Change in Military Total", ['President', 'House','Senate'])

def lin_reg(y, x_factors):
    X = df_data[x_factors]
    Y = df_data[y]
    results = sm.OLS(Y,sm.add_constant(X), missing = 'drop').fit()
    print(results.summary())

lin_reg(["Gov Marker"],["Change in Debt/GDP Ratio"])
lin_reg(["Gov Marker", "Year"],["True Approval"]) #indicative
lin_reg(["Gov Marker"],["Military Total"])

plt.figure()
plt.scatter(X,Y)

X_plot = np.linspace(0,1,100)
plt.plot(X_plot, X_plot*results.params[0] + results.params[1])

plt.show()

def cart(y, x_input):
    a = df_data[x_input+[y]].dropna()
    one_hot_data = pd.get_dummies(a[x_input], dummy_na=True)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(one_hot_data, a[y])
    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render("iris")
    tree.plot_tree(clf.fit(one_hot_data, a[y]))
    plt.show()

cart("Military Total", ['President', 'House', 'Senate'])


    # party_control.append([i, ])