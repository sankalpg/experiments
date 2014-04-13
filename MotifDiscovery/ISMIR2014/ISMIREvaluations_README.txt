This document has all the details neccessary to recall what I did for ISMIR 2014 motif paper, focusing on the evaluation aspect. 

I ALSO MENTION OBSERVATIONS DURING EXPERT FEEDBACK AND FUTURE DIRECTIONS AND HUNCHES!!


OVERVIEW

1) We performed seed motif discovery within song (details in the paper)
2) Taking seed patterns as query we searched for their 200 nearest patterns in the entire collection (details in the paper)
3) We refine rank of the searched patterns using 4 different distance measures.
3) All the patterns obtained are stored in a psql database named motifDB_CONF1. Also directories containing the audio files + features contain dump of everything. Including parameters used for processing.

Distance meaures used are squared Euclidean, city block, shifted city block and shifted cityblock till 100 cents and then exponential



STATS

Total number of files for which seeds exist:................1764 files
Total number of seeds obtained:.............................79172 (39586 pairs!)
Total number of searched patterns...........................15816400

Note that 15816400 / 200 ~= 79172 but 79082. This means for some files we were able to obtain seed patters but they couldn't successfully fetch any searched results. Will be looked into later (now no time)


EVALUATION:

We subsampled seed pattern space, dividing distance space into 3 classes. boundaties are mean(log(distance))+- std(log(distance)). We sampled 200 total patterns from this space such that every class has equal number of patterns (ofcourse with 1 extra pattern for class1 as 200%3=1).

Vignesh listened to 200*(10*4) = 8000 patterns and annotated each patterns with 
BAD = if the melodic similarity was not good
OK  = If the melodic similarity was good
GOOD = if the melodic similarity was good + they actually belonged to same musical entity (kind of a subjective decision I think).

For reporing in the paper we will consider both OK and GOOD as the same class and it becomes a binary decision then.

We call this expert feedback and made a web interface to assist the tedius task. All the files are checked-in in the same repository.


INTERESTING OBSERVATIONS / CASES (during annotations)

Cool Demo stuff, seed pattern id = 56, 58, 61, 64, 98, 106, 116, 159, 199, 175

FUNNY, seed pattern id = 181, 157

Seed motifs Ids for which a seed pattern was not in top 10 searched patterns: 6, 8, 23, 32, 47, 113 

Crappy audio Seed pattern id = 53, not noted but there were many, should have noted down.!



EVALUTION PURPOSE:
1) To see how seed motif distance relates to the perceptual melodic similarity.
2) IS there any clear threshold to separate perceptually similar pattern pairs from the others or there is a big overlap in  the distance distribution of two classes. The less overlap the better the distance measure.

3) To see how searched motif distance (computed using 4 different distance meaures) relates to the perceptual melodic similarity. Which of the distance measure aligns well with the perceptual melodic.
4) Is there is any clear threshold for determining perceptual similarity automatically. 
5) What is the overlap in distribution of distances of two clasess (similar and non similar) for all the rank refinement variants.
3) Is there any dependence of the type of seed pattern and the best method to find its repeated occuracnces (searching). Or the best method remains best for any kind of pattern or seed pair distance.
similarity.
4) Is there any correlation between seed pattern distance and the quality of the searched patterns or the amount of good patterns in searched patterns.
5) Can we determine if a seed pattern is meaningful looking at the searched data automatically.
6) What is the amount of mutually exclusive patterns that different rank refinement method have obtained in top 10 patterns.

EVALUATION MEASURES
1) Total # of similar patterns in top 10, for different seed pair category by different variants of rank refinement methods.
2) For the best rank refinement method what is the MAP, reciprocal rank. 

