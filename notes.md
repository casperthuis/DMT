# DMT - Assignment 1 - Notes

## Data

### Vars:

mood 	The mood scored by the user on a scale of 1-10
circumplex.arousal 	The arousal scored by the user, on a scale between -2 to 2
circumplex.valence 	The valence scored by the user, on a scale between -2 to 2
activity 	Activity score of the user (number between 0 and 1)
screen 	Duration of screen activity (time)
call 	Call made (indicated by a 1)
sms 	SMS sent (indicated by a 1)
appCat.builtin 	Duration of usage of builtin apps (time)
appCat.communication 	Duration of usage of communication apps (time)
appCat.entertainment 	Duration of usage of entertainment apps (time)
appCat.finance 	Duration of usage of finance apps (time)
appCat.game 	Duration of usage of game apps (time)
appCat.office 	Duration of usage of office apps (time)
appCat.other 	Duration of usage of other apps (time)
appCat.social 	Duration of usage of social apps (time)
appCat.travel 	Duration of usage of travel apps (time)
appCat.unknown 	Duration of usage of unknown apps (time)
appCat.utilities 	Duration of usage of utilities apps (time)
appCat.weather 	Duration of usage of weather apps (time)

### Patients

Total: 27

AS14.01    21999
AS14.23    21852
AS14.13    19592
AS14.28    19276
AS14.06    18092
AS14.29    17499
AS14.12    17311
AS14.30    17279
AS14.26    16403
AS14.33    16390
AS14.07    16045
AS14.17    15826
AS14.05    15745
AS14.02    14581
AS14.27    14575
AS14.24    14430
AS14.03    14425
AS14.25    12589
AS14.31    11889
AS14.19    11397
AS14.32    11193
AS14.09    10886
AS14.14     9286
AS14.08     7902
AS14.16     3982
AS14.20     3620
AS14.15     2848

## Algorithms

### Temporal: 

* ARIMA

### Non-temporal ('flattened'):

* SVM
* decision tree


## Steps

### Exploration

* Look at differences among patients (different variances over time?)
* 
* Look at corrolations in data
* Check for stationarity (http://www.maths.bris.ac.uk/~guy/Research/LSTS/TOS.html)
* 

### Preprocess

* Op dag niveau!
* Create dataset for non-temporal algorithms
	* Create table: per time interval, averages/sums of data
* Welke dag?
* Van ruwe metingen naar dag niveau
* 

### Train
* 


### Validation

* Bijv. alles vanaf 1 tijdspunt