## Description

This Crypto Dashboard displays real-time data of the ten most popular cryptocurrencies regarding the price and volume, as well as the predictions for the next three days and some additional information. It also includes real-time Twitter updates concerning posts and discussions related to cryptocurrencies. 

The dashboard is deployed on Render using Plotly’s open-source library Dash, and PyCharm as IDE. The data is updated every time the page is refreshed. 

## How to navigate through the dashboard? 

As most of the visualizations change for each cryptocurrency, the user first needs to select which cryptocurrency to display. The filter at the top of the page allows you to choose between the following ten currencies: ADA, ATOM, AVAX, AXS, BTC, ETH, LINK, LUNA1, MATIC, and SOL.

![Alt text](https://github.com/TJZLiu/Business-Case-5-Cryptocurrency-Dashboard/blob/main/readme_img/screenshot%201.png?raw=true "Optional Title")

By switching between different cryptocurrencies, most of the elements on the dashboard will adapt and display information only for that specific currency. 

![Alt text](https://github.com/TJZLiu/Business-Case-5-Cryptocurrency-Dashboard/blob/main/readme_img/screenshot%202.png?raw=true "Optional Title")

The main graph, the Candlestick with Rangeslider, displays the daily prices and volume for the selected cryptocurrency. By default, it shows the data for the last four months where each candlestick represents one day. However, there are two additional filters that the user can apply. Firstly, the user can choose what period the candlestick should represent: 1 Day, 3 Days, 1 Week, 2 Weeks, or 1 Month. Secondly, we included some financial indicators such as the Simple Moving Average (SMA), Exponential Moving Average (EMA), Moving Standard Deviation (MSTD), Moving Variance (MVAR), and Bollinger Bands (Bolling). By selecting one of the indicators, the user can see it displayed together with the candlestick graph.

![Alt text](https://github.com/TJZLiu/Business-Case-5-Cryptocurrency-Dashboard/blob/main/readme_img/screenshot%203.png?raw=true "Optional Title")

Hovering over a specific element of the graph gives the user more information. For the candlesticks, it shows information about the opening, highest, lowest, and closing price for that specific period, as well as indicating whether the price went up or down. Hovering over the volume tells the exact volume traded for that period. 

![Alt text](https://github.com/TJZLiu/Business-Case-5-Cryptocurrency-Dashboard/blob/main/readme_img/screenshot%204.png?raw=true "Optional Title")
![Alt text](https://github.com/TJZLiu/Business-Case-5-Cryptocurrency-Dashboard/blob/main/readme_img/screenshot%205.png?raw=true "Optional Title")

The Rangeslider also allows the user to choose the period it wants to observe. 

![Alt text](https://github.com/TJZLiu/Business-Case-5-Cryptocurrency-Dashboard/blob/main/readme_img/screenshot%206.png?raw=true "Optional Title")

Below the Candlestick graph, we can find the predictions for the next seven days based on two different time series models that we used in our previous project. The two selected models are the Holt Linear model and the Holt-Winters model. 

![Alt text](https://github.com/TJZLiu/Business-Case-5-Cryptocurrency-Dashboard/blob/main/readme_img/screenshot%207.png?raw=true "Optional Title")

On the left side of the above-mentioned graphs, the user finds some basic information about the selected cryptocurrency. Firstly, the logo and the full name of the chosen cryptocurrency are displayed. Below that, the user can find the trading information and the description of each cryptocurrency with the link to its website. 

![Alt text](https://github.com/TJZLiu/Business-Case-5-Cryptocurrency-Dashboard/blob/main/readme_img/screenshot%208.png?raw=true "Optional Title")

On the right side of the dashboard, the user can find a table that displays the overview of yesterday's prices for all other cryptocurrencies except the selected one. Below the table, there is a Twitter list, which includes tweets that are involved in a Bitcoin project or discuss Bitcoin regularly. This list is updated in real-time, displaying all new tweets immediately after they are posted. 

![Alt text](https://github.com/TJZLiu/Business-Case-5-Cryptocurrency-Dashboard/blob/main/readme_img/screenshot%209.png?raw=true "Optional Title")

The final visualization on the dashboard is a Choropleth Map, which tells us whether cryptocurrencies are legal in certain countries or not. By default, the user sees the status for the whole world, but there is also an option to filter based on the continents. 

![Alt text](https://github.com/TJZLiu/Business-Case-5-Cryptocurrency-Dashboard/blob/main/readme_img/screenshot%2010.png?raw=true "Optional Title")

Also, the user can hover over each country to see whether cryptocurrencies are legal or not in that country and see the country code. 
