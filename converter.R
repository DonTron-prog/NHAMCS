library(haven)

download.file(url = "ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/nhamcs/stata/ed2018-stata.zip",destfile = "ed2018-stata.zip")
unzip("ed2018-stata.zip")
nhamcs2018 <- read_dta("ED2018-stata.dta") 
write.csv(nhamcs2018,"/home/donald/github/NHAMCS/nhamcs2018.csv")
