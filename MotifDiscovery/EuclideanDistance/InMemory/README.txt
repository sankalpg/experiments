AIM: Can we obtain meaningful patterns in Carnatic Alaps using Euclidean distance measure (which looks tractable for large datasets). Specifically
1) Running SOTA methods which use Euclidean distance measures to handle large datasets
2) How does change in tempo (which might be expected across pieces) will affect Euclidean distance
3) How does slight shift in starting position while matching affects Euclidean distance
4) How does small improvisation (pitch and timing variations) within a pattern affects Euclidean distance
5) Can a big dataset help to alleviate these problems due to variabilities in pith, time (tempo + non linear variations)
6) Can some meaningless patterns have a low Euclidean distance than meaningful patterns (in large datasets)
7) What are the limitations of SOTA methods


Dataset:
Carnatic Alap IITM collection, 165 excerpts, 8 ragas. After silence removal over 3 million samples with 10 ms pitch hop.


Running SOTA METHODS:

###############################################################################
MK-Algorithm: EXACT MOTIF DISCOVERY
###############################################################################

Subsequence version:

Overall Problems:
1) Patterns with a lot of flat melody region in it. (SOLUTION: instead of providing big time series to the algorithm,  provide input in the form of subsequences, i.e. 'dataset' mode of the code in which I can remove subsequences which are not desired)  

2) Sometimes the patterns obtained are just rubbish. No meaningful shape but yet the distance is low (in relation to the actual desired patterns). I am still figuring it out why so.

3) I remove silence regions in the melody before processing, as a result sometimes the patterns obtained span melody regions across silence. Is this point clear? SOLUTION: instead of providing big time series to the algorithm,  provide input in the form of subsequences, i.e. 'dataset' mode and remove such subsequences.

3) Even through its an exact motif search, on multiple runs I obtain slightly different result.

4) At times I obtain result in 2 minutes but sometimes it takes like 10 minutes to obtain output. 

5) There is a normalization step in the code (subtracting mean and division by  std for every subsequence). I think since we want to be musically meaningful this division by std is incorrect in our case.


### Different Running Conditions ###

CODE:
Original CODE_ORIG
No normalization: CODE_NONORM

DATASET:
All alaps + original sampling rate + silence removed : DATA_ALL_DOWNSAMPLE1_SILENCEREM (HOP=10ms)
All alaps + downsample by 2 + silence removed : DATA_ALL_DOWNSAMPLE2_SILENCEREM (HOP=10ms)
All alaps + downsample by 3 + silence removed : DATA_ALL_DOWNSAMPLE3_SILENCEREM (HOP=10ms)


1) DATA_ALL_DOWNSAMPLE1_SILENCEREM + CODE_NONORM --> 6 million points and 3 second pattern (300) samples would need around 14gbs of ram
doesn't fit in the memory

2) DATA_ALL_DOWNSAMPLE2_SILENCEREM + CODE_NONORM --> 3 million points and 3 second pattern (150) samples
flat region as motif

3) DATA_ALL_DOWNSAMPLE2_SILENCEREM + CODE_NONORM --> 3 million points and 2 second pattern (100) samples
flat region as motif

Kambhoji alaps + original sampling rate + silence removed : KAMBHOJI_DOWNSAMPLE1_SILENCEREM (HOP=10ms)
Kambhoji alaps + downsample by 2 + silence removed : KAMBHOJI_DOWNSAMPLE2_SILENCEREM (HOP=10ms)
Kambhoji alaps + downsample by 3 + silence removed : KAMBHOJI_DOWNSAMPLE3_SILENCEREM (HOP=10ms)

4) KAMBHOJI_DOWNSAMPLE1_SILENCEREM + CODE_NONORM --> 1.2 MILLION points and 3 second pattern (300) samples
flat region as motif

5) KAMBHOJI_DOWNSAMPLE1_SILENCEREM + CODE_NONORM --> 1.2 MILLION points and 2 second pattern (200) samples
flat region as motif


Thodi CompMusic +  original sampling rate + silence removed : THODI_DOWNSAMPLE1_SILENCEREM (HOP=4.44ms)
Too big dataset

Thodi CompMusic +  downsample by 5 + silence removed : THODI_DOWNSAMPLE5_SILENCEREM (HOP=4.44ms)
Mostly flat regions

## After removing flat regions from the pitch data
Thodi CompMusic +  downsample by 5 + silence removed : THODI_DOWNSAMPLE5_SILENCEREM (HOP=4.44ms)
Decent patterns , though the system was slow

## Converted floating point to fix point and input data restricted from 0-600 and using 50-100 reference points instead of 10. Give a saving in memory and performance boost a bit
Decent patterns.






###############################################################################
Learnings/Conclusions/Remarks
###############################################################################







