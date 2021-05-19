library(haven)

year = "2011"

download.file(url = paste("ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/nhamcs/stata/ed", year, "-stata.zip", sep=''), destfile = paste("ed", year, "-stata.zip", sep=''))
unzip(paste("ed", year, "-stata.zip", sep=''))
nhamcs2018 <- read_dta(paste("ED", year, "-stata.dta", sep=''))
write.csv(nhamcs2018, paste("C:/Users/kvonk/python/NHAMCS/data/nhamcs", year, ".csv", sep=''))
