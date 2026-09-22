library(PMCMRplus) # provides frdAllPairsNemenyiTest()
library(reshape2)  # provides acast()
library(dplyr)     # provides recode_factor
library(tidyr)
library(stringr)
library(ggplot2)   # provides ggplot
library(effsize)   # provides cohen.d

# DATA PREPARATION
cols = c("data_type","alphabet_size","tier_size","language_class",
         "factor_width","threshold","index","split","accuracy","precision",
         "recall","f1","brier_score","model","train_size")
everything = read.csv('everything.csv', header=TRUE)[cols]

# Combine columns with a specific format
everything$train_setup <- sprintf("%s_%s", everything$train_size, everything$data_type)


##############################
#### Analysis by JH on FF ####
##############################

everything$data_type = as.factor(everything$data_type)
everything$language_class = as.factor(everything$language_class)
everything$split = as.factor(everything$split)
everything$index = as.factor(everything$index)
everything$model = as.factor(everything$model)
everything$train_size = as.factor(everything$train_size)
everything$train_setup = as.factor(everything$train_setup)

# rename some splits
everything$split <- recode_factor(everything$split, "SR" = "IR", "SA" = "IA")


nolarge <- everything %>% filter(train_size != "Large")

# SP: ok so the issue here was that copying directly had a variable name that 
# wasn't assigned in this notebook. Updated that. 

table1 <- nolarge %>%
  group_by(model, data_type, train_size) %>%
  summarize(meanaccuracy = round(mean(accuracy), 3))

table2 <- nolarge %>%
  filter(data_type == "OS") %>%
  group_by(model, split) %>%
  summarize(meanaccuracy = round(mean(accuracy), 3))

table3 <- nolarge %>%
  filter(data_type == "PS" & train_size == "Small") %>%
  group_by(model, split) %>%
  summarize(meanaccuracy = round(mean(accuracy), 3))

table4 <- nolarge %>%
  filter(data_type == "OL" & train_size == "Small") %>%
  group_by(model, split) %>%
  summarize(meanaccuracy = round(mean(accuracy), 3))


## Analysis begins here



# checking correlations among accuracy, Brier, and f-score measures
cor(nolarge[, c('accuracy', 'brier_score', 'f1')])


counts = read.table('counts.tsv', header=TRUE, sep='\t')

# Reporting basic stats on sizes of automata representations of the languages
# summary(counts$Size)
# summary(counts$Monoid)

# print(sd(counts$Size))
# print(sd(counts$Monoid))


data = merge(
  nolarge,
  counts,
  by.x=c("alphabet_size", "tier_size", "language_class", "factor_width", "threshold", "index"),
  by.y=c("Alph", "Tier", "Class", "k", "j", "i")
)



models = c('FF', 'simple', 'gru', 'lstm', 'transformer') 

md <- list()
for (m in models) {
   md[[m]] <- filter(data, model == m)
}

for (m in models) {
  print(m)
  
  print(cor(md[[m]][,c('accuracy', 'Size')]) )
  print(cor(md[[m]][,c('accuracy', 'Monoid')]) )
  print(cor(md[[m]][,c('accuracy', 'D.Classes')]) )
} 

#########################################################
#########################################################
# SET UP DATA FRAMES FOR FRIEDMAN TESTS
#########################################################
#########################################################

df <- list()
for (m in models) {
  df[[m]] <- aggregate(md[[m]]$accuracy,
               by=list(alphabet_size=md[[m]]$alphabet_size,
                       train_setup=md[[m]]$train_setup,
                       split=md[[m]]$split,
                       language_class=md[[m]]$language_class),
               FUN=mean)
  
}  

#########################################################
#########################################################
#########################################################


# ========================================================================
# FRIEDMAN TEST FOR TRAINING CONDITION
# ========================================================================

for (m in models) {
  print(m)
  data.matrix = acast(df[[m]],
                    alphabet_size + language_class + split ~ train_setup,
                    value.var="x")
  print(friedman.test(data.matrix))

  # POST HOC MULTIPLE COMPARISONS ANALYSIS
  print(frdAllPairsNemenyiTest(data.matrix))
  print(colMeans(data.matrix))
}



# ========================================================================
# FRIEDMAN TEST FOR TEST TYPES
# ========================================================================
# DOES NN ACCURACY DECREASE ACROSS THE TEST TYPES SR < (SA <> LR) < LA?

for (m in models) {
  print(m)
  data.matrix = acast(df[[m]],
                    alphabet_size + language_class + train_setup ~ split,
                    value.var="x")
  print(friedman.test(data.matrix))

  # POST HOC MULTIPLE COMPARISONS ANALYSIS
  # frdAllPairsNemenyiTest(data.matrix)
  print(colMeans(data.matrix))
}

# This gives the data in Tables 3,4 on page 9 and table 8,9,10 in the appendix.
setups = c('Mid_OL', 'Mid_PS', 'Small_OS', 'Small_PS', 'Small_OL')
for (m in models) {
  for (stp in setups) {
    cat(m, stp,"\n")
    data.matrix = acast(df[[m]][df[[m]]$train_setup == stp,],
                      alphabet_size + language_class + train_setup ~ split,
                      value.var="x")
  #print(friedman.test(data.matrix))
  
  # POST HOC MULTIPLE COMPARISONS ANALYSIS
  #print(frdAllPairsNemenyiTest(data.matrix))
  print(sort(colMeans(data.matrix)))
  cat("\n","\n")
  }
}



# ========================================================================

# FRIEDMAN TEST FOR LANGUAGE CLASSES
# ========================================================================

# Table 5 on page 10 comes from FF analysis in here:
for (m in models) {
  cat("\n", m, "\n")
  data.matrix = acast(df[[m]],
                    alphabet_size + split + train_setup ~ language_class,
                    value.var="x")
  
  print(friedman.test(data.matrix))
  print(frdAllPairsNemenyiTest(data.matrix))
  print(colMeans(data.matrix))
}

# looking at short strings only

for (m in models) {
  cat("\n", m, "\n")
  data.matrix = acast(df[[m]][df[[m]]$train_setup == 'Small_OS',],
                      alphabet_size + split + train_setup ~ language_class,
                      value.var="x")
  
  print(friedman.test(data.matrix))
  print(frdAllPairsNemenyiTest(data.matrix))
  print(colMeans(data.matrix))
}


# ========================================================================
# DIFFERENCES BETWEEN CNL, DPL, PROP, FO, REG
# ========================================================================

cnl = c("SL", "SP", "TSL")
dpl = c("coSL", "coSP", "TcoSL")
prop = c("LT", "PLT", "PT", "TLT", "TPLT")
fo = c("LTT", "TLTT", "SF")
reg = c("Zp", "Reg")
data.ff <- filter(data, model == m)
data.ff$logic <-
     ifelse(data.ff$language_class %in% cnl, "CNL",
     ifelse(data.ff$language_class %in% dpl, "DPL",
     ifelse(data.ff$language_class %in% prop, "PROP",
     ifelse(data.ff$language_class %in% fo, "FO",
     ifelse(data.ff$language_class %in% reg, "REG", "OTHER")))))


logic.agg = aggregate(data.ff$accuracy,
                      by=list(alphabet_size=data.ff$alphabet_size,
                              train_size=data.ff$train_size,
                              split=data.ff$split,
                              logic=data.ff$logic),
                      FUN=mean)
data.matrix = acast(logic.agg,
                    alphabet_size + train_size + split ~ logic,
                    value.var="x")

friedman.test(data.matrix)


# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
sort(colMeans(data.matrix))


# GRAPHS



# Box plots of the overall difference
temp <- data %>%
  filter(data_type == "PS" & train_size == "Small") %>%
  group_by(model, split) %>%
  summarize(meanaccuracy_ps = round(mean(accuracy), 3))

# Explore the difference in accuracy scores between PS on OL
# This is Figure 1 page 8 in the paper.
diff <- inner_join(temp, table4, by=c("model", "split"))
diff$accuracy_diff = diff$meanaccuracy_ps - diff$meanaccuracy
diff %>%
  ggplot(aes(x = model, y = accuracy_diff, fill = model)) +
  geom_col(alpha = 0.7) +
  facet_wrap(~factor(split, levels=c("SR", "SA", "LR", "LA"))) +
  theme_bw() +
  ylab("Difference in Accuracy") +
  xlab("") +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 35, vjust = 1, hjust = 1),
        legend.position = "none"
  ) +
  ggtitle("Difference in Acc from Small-PS to OI by Model")
ggsave("figs/acc-diff-ps-ol-box.pdf", width=6, height=4)
