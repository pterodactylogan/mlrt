library(PMCMRplus) # provides frdAllPairsNemenyiTest()
library(reshape2)  # provides acast()
library(dplyr)     # provides recode_factor
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

nolarge <- everything %>% filter(train_size != "Large")


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
# SET UP DATA FRAMES FOR FRIENMAN TESTS
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


# REG IS HARDEST TO LEARN. OTHERS ARE NOT SO CLEAR.
# ========================================================================
for (ntw in networks) {
    data.matrix = acast(logic.agg[logic.agg$network_type == ntw,],
                        alph + network_type + train_set_size + test_type ~ logic,
                        value.var="x")
    print(ntw)
    print(sort(colMeans(data.matrix)))
    print(friedman.test(data.matrix))
    # POST HOC MULTIPLE COMPARISONS ANALYSIS
    print(frdAllPairsNemenyiTest(data.matrix))
  }






# ========================================================================
# DIFFERENCES BETWEEN PROPOSITIONAL LOGICS BY ORDER RELATION
# ========================================================================
succ = c("SL", "coSL", "LT", "LTT")
prec = c("coSP", "PT", "SF", "SP")
tsucc = c("TcoSL", "TLT", "TLTT", "TSL")
data$prop <-
  ifelse(data$language_class %in% succ, "SUCC",
         ifelse(data$language_class %in% prec, "PREC",
                ifelse(data$language_class %in% tsucc, "TSUCC", "OTHER")))

prop.agg = aggregate(data$accuracy,
                     by=list(alphabet_size=data$alphabet_size,
                             train_size=data$train_size,
                             split=data$split,
                             prop=data$prop),
                     FUN=mean)
data.matrix = acast(prop.agg,
                    alphabet_size + train_size + split ~ prop,
                    value.var="x")

friedman.test(data.matrix)


# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
sort(colMeans(data.matrix))

# ========================================================================

for (ntw in networks) {
    data.matrix = acast(logic.agg[logic.agg$network_type == ntw,],
                        alph + network_type + train_set_size + test_type ~ prop,
                        value.var="x")
    print(ntw)
    print(sort(colMeans(data.matrix)))
    print(friedman.test(data.matrix))
    # POST HOC MULTIPLE COMPARISONS ANALYSIS
    print(frdAllPairsNemenyiTest(data.matrix))
  }




# ========================================================================
# FRIEDMAN TEST FOR ALPHABET SIZES:
# ========================================================================
data.matrix = acast(df,
                    language_class + train_size + split ~ alphabet_size,
                    value.var="x")
friedman.test(data.matrix)

# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
colMeans(data.matrix)

for (ntw in networks) {
  data.matrix = acast(df[df$network_type == ntw,],
                      class + network_type + train_set_size + test_type ~ alph,
                      value.var="x")
  print(ntw)
  print(sort(colMeans(data.matrix)))
  print(friedman.test(data.matrix))
  # POST HOC MULTIPLE COMPARISONS ANALYSIS
  print(frdAllPairsNemenyiTest(data.matrix))
}


# ========================================================================


# ========================================================================
# FRIEDMAN TEST FOR NETWORK TYPES
# ========================================================================
# ALL Training Sets
data.matrix = acast(df,
                    alph + class + train_set_size + test_type ~ network_type,
                    value.var="x")
friedman.test(data.matrix)


# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
sort(colMeans(data.matrix))

# ========================================================================

# Small Training Set
# ==================
df.temp = df[df$train_set_size == "Small",]
data.matrix = acast(df.temp,
                    alph + class + train_set_size + test_type ~ network_type,
                    value.var="x")
friedman.test(data.matrix)


# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
colMeans(data.matrix)

# ========================================================================



# Mid Training Set
df.temp = df[df$train_set_size == "Mid",]
data.matrix = acast(df.temp,
                    alph + class + train_set_size + test_type ~ network_type,
                    value.var="x")
friedman.test(data.matrix)


# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
colMeans(data.matrix)

# ========================================================================


# Large Training Set
df.temp = df[df$train_set_size == "Large",]
data.matrix = acast(df.temp,
                    alph + class + train_set_size + test_type ~ network_type,
                    value.var="x")
friedman.test(data.matrix)

# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
colMeans(data.matrix)

# ========================================================================



# DO k VALUES MAKE A DIFFERENCE FOR PROP1 AND PROP2?
# ========================================================================
succ = c("SL", "coSL", "LT", "LTT")
prec = c("coSP", "PT", "SP") # removed SF
tsucc = c("TcoSL", "TLT", "TLTT", "TSL")

prop1 = c(succ, prec, tsucc)
prop2 = prop1[!(prop1 %in% c("LTT", "TLTT"))]

df.temp = aggregate(data$accuracy,
                    by=list(alph=data$alph,
                            class=data$class,
                            k=data$k,
                            network_type=data$network_type,
                            train_set_size=data$train_set_size,
                            test_type=data$test_type),
                    FUN=mean)

df.temp1 = df.temp[df.temp$class %in% prop1,]
data.prop1 = acast(df.temp1,
                   alph + class + network_type + train_set_size + test_type ~ k,
                   value.var="x")
df.temp2 = df.temp[df.temp$class %in% prop2,]
data.prop2 = acast(df.temp2,
                   alph + class + network_type + train_set_size + test_type ~ k,
                   value.var="x")

friedman.test(data.prop1)

# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.prop1)
colMeans(data.prop1)


friedman.test(data.prop2)

# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.prop2)
colMeans(data.prop2)

# ========================================================================



# ========================================================================
# VISUALIZATIONS
# ========================================================================

lang.order = c("SL", "coSL", "TSL", "TcoSL", "SP", "coSP", "LT", "TLT",
               "PT", "LTT", "TLTT", "PLT", "TPLT", "SF", "Zp", "Reg")
size.order = c("Small", "Mid", "Large")
test.order = c("SR", "LR", "SA", "LA")
nn.order = c("Simple RNN", "GRU", "LSTM", "2-layer LSTM", "Transformer")
drc.order = c("1Q", "2Q", "3Q", "4Q")

( # VISUALIZE CAST.DF, i.e. CLASS ACCURACY AGGREGATED OVER TIER, k, j, i
  ggplot(stack(cast.df), aes(x=ind, y=values))
  + geom_boxplot()
  + scale_x_discrete(limits=lang.order)
  + ggtitle("Accuracy by Class")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="Language Class", y="Accuracy")
)

jpeg("acc_class_alph.jpeg", units="in", width=10, height=5, res=300)
(
  ggplot(data, aes(x=class, y=accuracy, fill=alph))
  + geom_boxplot(outlier.shape=NA)
  + scale_x_discrete(limits=lang.order)
  + ggtitle("Accuracy by Class and Alphabet Size")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="Language Class", y="Accuracy", fill="Alphabet Size")
)
dev.off()

jpeg("acc_k_alph.jpeg", units="in", width=10, height=5, res=300)
( # ACCURACY BY k VALUES
  ggplot(data, aes(x=factor(k), y=accuracy, fill=alph))
  + geom_boxplot(outlier.shape=NA)
  + ggtitle("Accuracy by k Values and Alphabet Size")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="k Values", y="Accuracy", fill="Alphabet Size")
)
dev.off()

jpeg("acc_class_trainsize.jpeg", units="in", width=10, height=5, res=300)
(
  ggplot(data, aes(x=class, y=accuracy, fill=factor(train_set_size, levels=size.order)))
  + geom_boxplot(outlier.shape=NA)
  + scale_fill_discrete(limits=size.order)
  + scale_x_discrete(limits=lang.order)
  + ggtitle("Accuracy by Class and Training Set Size")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="Language Class", y="Accuracy", fill="Training Set Size")
)
dev.off()

jpeg("acc_class_test.jpeg", units="in", width=10, height=5, res=300)
(
  ggplot(data, aes(x=class, y=accuracy, fill=factor(test_type, levels=test.order)))
  + geom_boxplot(outlier.shape=NA)
  + scale_fill_discrete(limits=test.order)
  + scale_x_discrete(limits=lang.order)
  + ggtitle("Accuracy by Class and Test Type")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="Language Class", y="Accuracy", fill="Test Type")
)
dev.off()

jpeg("acc_class_test_large.jpeg", units="in", width=10, height=5, res=300)
(
  ggplot(data[data$train_set_size=="Large",], aes(x=class, y=accuracy, fill=factor(test_type, levels=test.order)))
  + geom_boxplot(outlier.shape=NA)
  + scale_fill_discrete(limits=test.order)
  + scale_x_discrete(limits=lang.order)
  + ggtitle("Accuracy by Class and Test Type")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="Language Class", y="Accuracy", fill="Test Type")
)
dev.off()


jpeg("acc_class_nn.jpeg", units="in", width=12, height=5, res=300)
(
  ggplot(data, aes(x=class, y=accuracy, fill=factor(network_type, levels=nn.order)))
  + geom_boxplot(outlier.shape=NA)
  + scale_fill_discrete(limits=nn.order)
  + scale_x_discrete(limits=lang.order)
  + ggtitle("Accuracy by Class and Network Type")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="Language Class", y="Accuracy", fill="Network Type")
)
dev.off()

jpeg("acc_class_nn_large.jpeg", units="in", width=12, height=5, res=300)
(
  ggplot(data[data$train_set_size=="Large",], aes(x=class, y=accuracy, fill=factor(network_type, levels=nn.order)))
  + geom_boxplot(outlier.shape=NA)
  + scale_fill_discrete(limits=nn.order)
  + scale_x_discrete(limits=lang.order)
  + ggtitle("Accuracy by Class and Network Type (Large Train Set Size)")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="Language Class", y="Accuracy", fill="Network Type")
)
dev.off()


#
# EXTRAS
#

# CREATE AGGREGATE MATRIX FOR NNs VS LANGUAGE CLASSES

trainsizes <- c('Small', 'Mid', 'Large')

for (size in trainsizes) {

  df.tmp = data[data$train_set_size == size,]
  agg.tmp = aggregate(df.tmp$accuracy,
                      by=list(network_type=df.tmp$network_type,
                              class=df.tmp$class),
                      FUN=mean)
  agg = acast(agg.tmp, network_type ~ class, value.var="x")
  print(size)
  print(agg)
}
