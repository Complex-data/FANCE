# FAke News Classification and Explanation framework (Fance)

## Installation
Install all the libraries in environment.yml using the following command:

    conda env create --file environment.yml

## Dataset
We deploy our experiments on two datasets of fake news detection, which are collected from FakeNewsNet(https://github.com/KaiDMML/FakeNewsNet).

### Dataset Structure
The downloaded dataset will have the following  folder structure,
```bash
├── gossipcop
│   ├── fake
│   │   ├── gossipcop-1
│   │	│	├── news content.json
│   │	│	├── tweets
│   │	│	│	├── 886941526458347521.json
│   │	│	│	├── 887096424105627648.json
│   │	│	│	└── ....		
│   │	│  	└── retweets
│   │	│		├── 887096424105627648.json
│   │	│		├── 887096424105627648.json
│   │	│		└── ....
│   │	└── ....			
│   └── real
│      ├── gossipcop-1
│      │	├── news content.json
│      │	├── tweets
│      │	└── retweets
│		└── ....		
├── politifact
│   ├── fake
│   │   ├── politifact-1
│   │   │	├── news content.json
│   │   │	├── tweets
│   │   │	└── retweets
│   │	└── ....		
│   │
│   └── real
│      ├── poliifact-2
│      │	├── news content.json
│      │	├── tweets
│      │	└── retweets
│      └── ....					
├── user_profiles
│		├── 374136824.json
│		├── 937649414600101889.json
│   		└── ....
├── user_timeline_tweets
│		├── 374136824.json
│		├── 937649414600101889.json
│	   	└── ....
└── user_followers
│		├── 374136824.json
│		├── 937649414600101889.json
│	   	└── ....
└──user_following
        	├── 374136824.json
		├── 937649414600101889.json
	   	└── ....
```


## Model Training
    python go_FANCE.py -xx xxx
optional arguments:


    -cc max comment count	int
    -cl max comment length	int
    -sc max sentence count	int
    -sl max sentence length	int
    -lr learning rate		int
    -bs batch size			int
    -pf platform	
ps: Except -pf, other parameters can only provide one modification at a time.		