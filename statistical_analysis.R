rm(list = ls())
library(pROC)
library(ggplot2)
library(ggalt)
library(dplyr)
library(caret)

setwd('F:/DKI/final_results/R')
#import results
clinical_data <- read.csv('F:/DKI/final_results/data_cohorts/clinical_data.csv')
clinical_data <- clinical_data[clinical_data$Cohort != 2, ]
clinical_data$Age <- ifelse(clinical_data$Age > 50, 1, 0)
clinical_data$Age <- factor(clinical_data$Age,
                             levels = c(1, 0),
                             labels = c(">50 years", "≤50 years"))
clinical_data$Cohort <- factor(clinical_data$Cohort, levels = c(1,3,4), 
                         labels = c('Training Set', 
                                    'Internal Testing Set', 
                                    'External Testing Set'))
clinical_data$Gender <- factor(clinical_data$Gender, levels = c(1,0),
                      labels = c('Male', 'Female'))
clinical_data$CEA <- factor(clinical_data$CEA, levels = c(1,0),
                               labels = c('>5.0ng/ml', '≤5.0ng/ml'))
clinical_data$CA199 <- factor(clinical_data$CA199, levels = c(1,0),
                               labels = c('>37.0U/ml', '≤37.0U/ml'))
clinical_data$Location <- factor(clinical_data$Location,
                                 levels = c(1,2,3),
                           labels = c('<5cm', '5~10cm', '>10cm'))
clinical_data$cTNM <- ifelse(clinical_data$cTNM > 3, 1, 0)
clinical_data$cTNM <- factor(clinical_data$cTNM, levels = c(1,0),
                             labels = c('III', 'II'))
clinical_data$cT <- ifelse(clinical_data$cT > 2, 1, 0)
clinical_data$cT <- factor(clinical_data$cT, levels = c(1,0),
                             labels = c('cT3-4', 'cT2'))
# clinical_data$cT <- factor(clinical_data$cT, levels = c(2,3,4,5),
#                              labels = c('T2', 'T3', 'T4a', 'T4b'))
clinical_data$cN <- ifelse(clinical_data$cN > 0, 1, 0)
clinical_data$cN <- factor(clinical_data$cN, levels = c(1,0),
                           labels = c('cN1-2', 'cN0'))
# clinical_data$cN <- factor(clinical_data$cN, levels = c(0,1,2),
#                          labels = c('N0', 'N1', 'N2'))
clinical_data$MRF <- factor(clinical_data$MRF, levels = c(1,0),
                             labels = c('+', '-'))
clinical_data$EMVI <- factor(clinical_data$EMVI, levels = c(1,0),
                             labels = c('+', '-'))
clinical_data$Differentiation <- factor(clinical_data$Differentiation,
                                        levels = c(1,2,3),
                                        labels = c('Well',
                                                   'Moderate',
                                                   'Poor'))
clinical_data$pT <- ifelse(clinical_data$pT > 4, 4, clinical_data$pT)
clinical_data$pT <- factor(clinical_data$pT, levels = c(0,1,2,3,4),
                           labels = c('T0','T1', 'T2',
                                      'T3', 'T4'))
# clinical_data$pT <- factor(clinical_data$pT, levels = c(0,1,2,3,4,5),
#                            labels = c('T0','T1', 'T2',
#                                       'T3', 'T4a','T4b'))
clinical_data$pN <- factor(clinical_data$pN, levels = c(0,1,2),
                           labels = c('N0', 'N1', 'N2'))
clinical_data$pLVI <- factor(clinical_data$pLVI, levels = c(1,0),
                       labels = c('+', '-'))
clinical_data$pPNI <- factor(clinical_data$pPNI, levels = c(1,0),
                      labels = c('+', '-'))
clinical_data$AC <- factor(clinical_data$AC, levels = c(1,0),
                     labels = c('+', '-'))
clinical_data$TRG <- factor(clinical_data$TRG, levels = c(0,1,2,3),
                            labels = c('0', '1', '2', '3'))
clinical_data$censorship <- factor(clinical_data$censorship, 
                                   levels = c(1,0),labels = c('+', '-'))



###########Clinical Characteristic##########
library(gtsummary)
library(flextable)
library(xtable)
library(officer)
library(coin)
library(tidyr)
baseline_data <- clinical_data[-c(1,2)]
baseline_table <- baseline_data %>%
  tbl_summary(
    by = Cohort,
    type = list(all_continuous() ~ "continuous",
                all_categorical() ~ "categorical"),
    statistic = all_continuous() ~ c( "{median} ({p25}, {p75})"),
    missing_text ='Missing',
    label=list(Differentiation~'Tumor Differentiation',
               censorship~'Recurrence or Metastases',
               DFS~'DFS Time (mouth)'),
    digits = all_continuous() ~ 0) %>%
  add_p(pvalue_fun = ~style_pvalue(.x, digits = 3),
        list(Location ~ "kruskal.test",
             pT ~ "kruskal.test",
             pN ~ "kruskal.test",
             TRG ~"kruskal.test",
             Differentiation ~ "kruskal.test"))%>%
  modify_caption("Patient Characteristics") 
baseline_table
baseline_table %>%
  as_flex_table() %>%
  flextable::save_as_docx(baseline_table,path='F:/DKI/final_results/R/baseline_table.docx')

library(rcompanion)
clinical_data <- read.csv('F:/DKI/final_results/data_cohorts/clinical_data.csv')
clinical_data <- clinical_data[clinical_data$Cohort != 2, ]
clinical_data$Cohort <- factor(clinical_data$Cohort, levels = c(1,3,4), 
                               labels = c('Training Set', 
                                          'Internal Test Set', 
                                          'External Test Set'))
kruskal_variables <- c("Location", "pT", "pN")
for (variable in kruskal_variables) {
  # 进行两两比较
  posthoc_result <- pairwise.wilcox.test(clinical_data[[variable]],
                                         clinical_data$Cohort, 
                                         p.adjust.method = "bonferroni")
  print(variable)
  print(posthoc_result)
  
  cat("\n")
}

############堆积柱状图#######################
library(plyr)
library(tidyverse)
library(readxl)
library(cowplot)
library(ggsci)
new_clinical_data <- subset(clinical_data, select = -c(id_name, center, DFS))
column_names <- colnames(new_clinical_data)
column_names <- column_names[-1]

plot_list <- list()
for (column_name in column_names) {
  variable <- column_name
  expr <- sym(variable)
  result <- new_clinical_data %>%
    group_by(Cohort, !!expr) %>%
    summarize(count = n()) %>%
    group_by(Cohort) %>%
    mutate(percentage = count / sum(count) * 100)
  
  p <- ggplot(result)+
    geom_bar(aes(Cohort, percentage, fill = !!expr), color = "#f3f4f4",
             position = "fill", stat = "identity", linewidth = 1)+
    xlab("")+
    ylab("")+
    theme_bw()+
    theme(axis.title.x = element_text(face = "bold"),
          axis.title.y = element_text(face = "bold"),
          panel.grid = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, face = "italic"),
          panel.background = element_rect(fill = "#f3f4f4"),
          legend.position = "top",
          plot.margin = unit(c(0, 0.5, 0, 0), "cm") 
    )+ scale_fill_npg() +
    scale_x_discrete(labels = c("Train", "Internal Test", "External Test"))
  plot_list[[column_name]] <- p
}

# 拼接所有图像
combined_plot <- plot_grid(plotlist = plot_list, ncol = 6) # Adjust the plot margins as needed

# 打印拼接后的图像
print(combined_plot)

ggsave('F:/DKI/final_results/plot/clinical_barplot.tiff', plot = combined_plot, 
       device =  "tiff", width = 6500,height = 3000,units = 'px',dpi = 300)


# 创建一个包含变量名称的向量
chisq_variables <- c("MRF", "EMVI", "pLVI", "AC")
# 打开一个 TXT 文件用于写入结果
file_name <- "chisq_results.txt"
file_conn <- file(file_name, "w")
# 循环遍历每个变量
for (variable in chisq_variables) {
  # 创建列联表
  tab <- table(baseline_data$Cohort, baseline_data[[variable]])
  print(variable)
  # 进行两两比较
  pairwise_result <- pairwise.prop.test(tab, p.adjust.method = "bonferroni")
  print(pairwise_result)
  cat("\n")
}

#################dataimport##################
library(timeROC)
library(survival)
library(survivalROC)
library(readxl)
library(ggplot2)
library(ggpubr)
library(pROC)
library(rms)
library(pec)

result_paths <- c("F:/DKI/final_results/excel_res/internal_test_results.csv",
                  "F:/DKI/final_results/excel_res/external_test_results.csv")
cohort_names <- c("Internal Testing Set", "External Testing Set")
predictors <- c('all_CLS_pre', 'cli_CLS_pre', 'precli_CLS_pre',
                't2_CLS_pre', 'img_CLS_pre', 'img_precli_CLS_pre')
model_names <- c("All Model", "Clinical Model", "Preclinical Model",
                 "T2 Model", "T2+DKI Model", "T2+DKI+Preclinical Model")
######################ROC########################################################
for (i in seq_along(result_paths)){
  result_path <- result_paths[i]
  cohort_name <- cohort_names[i]
  #############import Result###############
  test_result <- read.csv(result_path, header = TRUE, row.names = 1)
  # 显示图形
  img_roc = roc(CLS_GT~img_CLS_pre, smooth = T,smooth.method="binormal",
                percent = TRUE, data = test_result, auc= TRUE, plot = FALSE)
  t2_roc = roc(CLS_GT~t2_CLS_pre, smooth = T,smooth.method="binormal",
               percent = TRUE, data = test_result, auc= TRUE, plot = FALSE)
  img_precli_roc = roc(CLS_GT~img_precli_CLS_pre, smooth = T,smooth.method="binormal",
                       percent = TRUE, data = test_result, auc= TRUE, plot = FALSE)
  all_roc = roc(CLS_GT~all_CLS_pre, smooth = T,smooth.method="binormal",
                percent = TRUE, data = test_result, auc= TRUE, plot = FALSE)
  cli_roc = roc(CLS_GT~cli_CLS_pre, smooth = T,smooth.method="binormal",
                percent = TRUE, data = test_result, auc= TRUE, plot = FALSE)
  precli_roc = roc(CLS_GT~precli_CLS_pre, smooth = T,smooth.method="binormal",
                   percent = TRUE, data = test_result, auc= TRUE, plot = FALSE)

  auc_values <- c(auc(all_roc)/100, auc(cli_roc)/100, auc(precli_roc)/100,
                  auc(t2_roc)/100, auc(img_roc)/100, auc(img_precli_roc)/100)
  low_ci_values <- c(ci.auc(all_roc)[1]/100, ci.auc(cli_roc)[1]/100,
                     ci.auc(precli_roc)[1]/100, ci.auc(t2_roc)[1]/100,
                     ci.auc(img_roc)[1]/100, ci.auc(img_precli_roc)[1]/100)
  up_ci_values <- c(ci.auc(all_roc)[3]/100, ci.auc(cli_roc)[3]/100,
                    ci.auc(precli_roc)[3]/100, ci.auc(t2_roc)[3]/100,
                    ci.auc(img_roc)[3]/100, ci.auc(img_precli_roc)[3]/100)
  auc_labels <- sprintf("%s: %.2f (%.2f-%.2f)", model_names, auc_values,
                        low_ci_values,up_ci_values)
  auc_df <- data.frame(Model = model_names,
                       AUC = auc_values,
                       AUC_Lower_CI = low_ci_values,
                       AUC_Upper_CI = up_ci_values,
                       stringsAsFactors = FALSE)
  roc_data <- data.frame(
    FPR = c(all_roc$specificities, cli_roc$specificities, precli_roc$specificities,
            t2_roc$specificities, img_roc$specificities, img_precli_roc$specificities),
    TPR = c(all_roc$sensitivities, cli_roc$sensitivities, precli_roc$sensitivities,
            t2_roc$sensitivities, img_roc$sensitivities, img_precli_roc$sensitivities),
    Model = c(rep(auc_labels[1], length(all_roc$specificities)),
              rep(auc_labels[2], length(cli_roc$specificities)),
              rep(auc_labels[3], length(precli_roc$specificities)),
               rep(auc_labels[4], length(t2_roc$specificities)),
               rep(auc_labels[5], length(img_roc$specificities)),
               rep(auc_labels[6], length(img_precli_roc$specificities)))
    )
  roc_data <- arrange(roc_data, TPR, FPR)

  # 绘制ROC曲线和标签
  ROC_curves <- ggplot(roc_data, aes(x = 100-FPR, y = TPR, color = Model)) +
    geom_line(aes(group = Model), size = 1.5) +
    geom_abline(slope = 1, intercept = 0, color = "grey", size = 1, linetype = 2) +
    theme_bw() +
    labs(x = "1-False positive rate(%)", y = "True positive rate(%)") +
    theme(
      axis.text = element_text(face = "bold", size = 11, color = "black"),
      axis.title.x = element_text(face = "bold", size = 14, color = "black"),
      axis.title.y = element_text(face = "bold", size = 14, color = "black"),
      legend.position = c(0.7,0.15), legend.background = element_blank(),
      legend.box.background = element_rect(fill = NA,
                                           colour = 'black',
                                           linetype = 1,
                                           size = 1.5),
      legend.text = element_text(size = 20),
      legend.title = element_text(size = 18, face = 'bold'),
      legend.margin = margin(10, 10, 10, 10), # 调整图例的边距
      legend.spacing.y = unit(0.5,'cm')
    )  + scale_color_manual(values = c("#E64B35", "#4DBBD5", "#E6C02B",
                                       "#147A5C", "#8700FF", "#800000"),
                            name = 'AUC of each model'
    )+ scale_x_continuous(expand=c(0, 0),limits = c(0,100)) +
    scale_y_continuous(expand=c(0, 0),limits = c(0,100))
  ROC_curves
  save_path = paste0('F:/DKI/final_results/plot/', cohort_name,'_ROC.tiff')
  ggsave(save_path, plot = ROC_curves,
          width = 3900,height = 3000,units = 'px',dpi = 300)
  ##############delong test#####################
  # img_roc = roc(CLS_GT~img_CLS_pre, data = test_result)
  # t2_roc = roc(CLS_GT~t2_CLS_pre, data = test_result)
  # img_precli_roc = roc(CLS_GT~img_precli_CLS_pre, data = test_result)
  # all_roc = roc(CLS_GT~all_CLS_pre, data = test_result)
  # cli_roc = roc(CLS_GT~cli_CLS_pre, data = test_result)
  # precli_roc = roc(CLS_GT~precli_CLS_pre, data = test_result)
  # print(cohort_name)
  # delong_result <- roc.test(t2_roc, img_roc, boot.n = 1000, method = "delong",
  #                           alternative = "less")
  # print(delong_result)
  # delong_result <- roc.test(precli_roc, img_roc, boot.n = 1000, method = "delong",
  #                           alternative = "less")
  # print(delong_result)
  # delong_result <- roc.test(img_roc, img_precli_roc, boot.n = 1000, method = "delong",
  #                           alternative = "less")
  # print(delong_result)
  # delong_result <- roc.test(img_precli_roc, all_roc, boot.n = 1000, method = "delong",
  #                           alternative = "less")
  # print(delong_result)
  # delong_result <- roc.test(cli_roc, all_roc, boot.n = 1000, method = "delong",
  #                           alternative = "less")
  # print(delong_result)
  #######雷达图##########
  # library(tidyverse)
  # min_break = floor(min(auc_df$AUC))
  # max_break = ceiling(max(auc_df$AUC))
  # mid_break = mean(c(min_break,max_break))
  # radar_plot <- ggplot(auc_df) +
  #   geom_hline(data = data.frame(y = c(min_break,mid_break,max_break)),
  #              aes(yintercept = y),color = "lightgrey") + 
  #   geom_col(aes(x = reorder(Model, AUC),y = AUC,fill = AUC),
  #            position = "dodge2",show.legend = TRUE,alpha = .9) +
  #   geom_segment(aes(x = reorder(Model, AUC),y = min_break,
  #                    xend = reorder(Model, AUC),yend = max_break),
  #                linetype = "dashed",color = "gray12") + 
  #   coord_polar()+
  #   theme_bw()
  # k = nrow(auc_df)+0.7
  # radar_plot <- radar_plot +
  #   annotate(x = k, y = min_break, label = min_break, geom = "text", color = "gray12") +
  #   annotate(x = k, y = mid_break, label = mid_break, geom = "text", color = "gray12") +
  #   annotate(x = k, y = max_break, label = max_break, geom = "text", color = "gray12") +
  #   scale_y_continuous(limits = c(-mid_break, max_break),
  #                      breaks = c(0, min_break, mid_break, max_break),expand = c(0, 0)) + 
  #   scale_fill_gradientn("AUC",colours = c( "#FFEDA0","#FEB24C","#FC4E2A","#800026")) +
  #   guides(fill = guide_colorsteps(barwidth = 15, barheight = .5, 
  #                                  title.position = "top", title.hjust = .5)) +
  #   theme(axis.title = element_blank(),axis.ticks = element_blank(),
  #         axis.text.y = element_text(face = "bold"),
  #         axis.text.y = element_blank(),legend.position = "bottom")
  # print(radar_plot)
  # save_path = paste0('F:/DKI/final_results/plot/', cohort_name, '_AUC_RadarPlot.tiff')
  # ggsave(save_path, plot = radar_plot,
  #         width = 3900,height = 3000,units = 'px',dpi = 300)
}

#ACC,SEN,SPE,PPV,NPV,RECALL,JACCARD
calculate_metrics <- function(predicted_labels, true_labels) {
  # 创建混淆矩阵
  confusion_matrix <- table(predicted_labels, true_labels)
  
  # 计算准确率
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  
  # 计算灵敏度（召回率）
  sensitivity <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
  
  # 计算特异度
  specificity <- confusion_matrix[1, 1] / sum(confusion_matrix[, 1])
  
  # 计算阳性预测值（PPV）
  ppv <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
  
  # 计算阴性预测值（NPV）
  npv <- confusion_matrix[1, 1] / sum(confusion_matrix[1, ])
  
  # 计算Jaccard指数
  jaccard <- confusion_matrix[2, 2] / (confusion_matrix[2, 2] 
                                       + confusion_matrix[1, 2] 
                                       + confusion_matrix[2, 1])
  
  # 创建结果表格
  result_table <- data.frame(Matrix = c("Accuracy", "Sensitivity", "Specificity",
                                        "PPV", "NPV", "Jaccard"),
                             Value = c(accuracy, sensitivity, specificity, 
                                       ppv, npv, jaccard))

  return(result_table)
}
result_table <- data.frame(Cohort = character(),
                           Model = character(),
                           Accuracy = numeric(), 
                           Sensitivity = numeric(), 
                           Specificity = numeric(), 
                           PPV = numeric(), 
                           NPV = numeric(),
                           Jaccard = numeric(), 
                           stringsAsFactors = FALSE)
for (i in seq_along(result_paths)){
  result_path <- result_paths[i]
  cohort_name <- cohort_names[i]
  #############import Result###############
  test_result <- read.csv(result_path, header = TRUE, row.names = 1)
  for (j in seq_along(predictors)) {
    predictor <- predictors[j]
    model_name <- model_names[j]
    # 创建模型
    test_result$CLS_pre <- ifelse(test_result[[predictor]] > 0.5, 1, 0)
  # 调用calculate_metrics函数，并将结果保存在result变量中
    metrics_result <- calculate_metrics(test_result$CLS_pre,
                                        test_result$CLS_GT)
    # 创建临时数据框，用于存储计算结果和对应的Cohort和Model值
    temp_df <- data.frame(Cohort = cohort_name,
                          Model = model_name,
                          metrics_result)
    # 将计算结果添加到结果表格中
    result_table <- rbind(result_table, temp_df)
  }
}
wide_table <- result_table %>%
  pivot_wider(names_from = Matrix, values_from = Value)

wide_table[, -c(1:2)] <- round(wide_table[, -c(1:2)], 3)

# 创建一个空的word_doc对象
doc <- read_docx()

# 创建flextable对象
table <- flextable(wide_table)

# 将表格添加到word_doc对象
doc <- body_add_flextable(doc, table)

# 保存Word文档
print(doc, target = 'F:/DKI/final_results/R/results_table.docx')


#train ROC
train_result <- read.csv("F:/DKI/final_results/excel_res/train_results.csv",
                         header = TRUE, row.names = 1)
img_roc = roc(CLS_GT~img_CLS_pre, data = train_result, auc= TRUE)
t2_roc = roc(CLS_GT~t2_CLS_pre, data = train_result, auc= TRUE)
img_precli_roc = roc(CLS_GT~img_precli_CLS_pre, data = train_result, auc= TRUE)
all_roc = roc(CLS_GT~all_CLS_pre, data = train_result, auc= TRUE)
cli_roc = roc(CLS_GT~cli_CLS_pre, data = train_result, auc= TRUE)
precli_roc = roc(CLS_GT~precli_CLS_pre, data = train_result, auc= TRUE)

auc_values <- c(auc(all_roc), auc(cli_roc), auc(precli_roc),
                auc(t2_roc), auc(img_roc), auc(img_precli_roc))
low_ci_values <- c(ci.auc(all_roc)[1], ci.auc(cli_roc)[1],
                   ci.auc(precli_roc)[1], ci.auc(t2_roc)[1],
                   ci.auc(img_roc)[1], ci.auc(img_precli_roc)[1])
up_ci_values <- c(ci.auc(all_roc)[3], ci.auc(cli_roc)[3],
                  ci.auc(precli_roc)[3], ci.auc(t2_roc)[3],
                  ci.auc(img_roc)[3], ci.auc(img_precli_roc)[3])
auc_labels <- sprintf("%s: %.2f (%.2f-%.2f)", model_names, auc_values,
                      low_ci_values,up_ci_values)
auc_df <- data.frame(Model = model_names,
                     AUC = auc_values,
                     AUC_Lower_CI = low_ci_values,
                     AUC_Upper_CI = up_ci_values,
                     stringsAsFactors = FALSE)

roc_data <- data.frame(
  FPR = c(all_roc$specificities, cli_roc$specificities, precli_roc$specificities,
          t2_roc$specificities, img_roc$specificities, img_precli_roc$specificities),
  TPR = c(all_roc$sensitivities, cli_roc$sensitivities, precli_roc$sensitivities,
          t2_roc$sensitivities, img_roc$sensitivities, img_precli_roc$sensitivities),
  Model = c(rep(auc_labels[1], length(all_roc$specificities)),
            rep(auc_labels[2], length(cli_roc$specificities)),
            rep(auc_labels[3], length(precli_roc$specificities)),
            rep(auc_labels[4], length(t2_roc$specificities)),
            rep(auc_labels[5], length(img_roc$specificities)),
            rep(auc_labels[6], length(img_precli_roc$specificities)))
)
roc_data <- arrange(roc_data, TPR, FPR)

ROC_curves <- ggplot(roc_data, aes(x = 1-FPR, y = TPR, color = Model)) +
  geom_line(aes(group = Model), size = 1.5) +
  geom_abline(slope = 1, intercept = 0, color = "grey", size = 1, linetype = 2) +
  theme_bw() +
  labs(x = "1-False positive rate(%)", y = "True positive rate(%)") +
  theme(
    axis.text = element_text(face = "bold", size = 11, color = "black"),
    axis.title.x = element_text(face = "bold", size = 14, color = "black"),
    axis.title.y = element_text(face = "bold", size = 14, color = "black"),
    legend.position = c(0.7,0.15), legend.background = element_blank(),
    legend.box.background = element_rect(fill = NA, 
                                         colour = 'black', 
                                         linetype = 1, 
                                         size = 1.5),
    legend.text = element_text(size = 20),
    legend.title = element_text(size = 18, face = 'bold'),
    legend.margin = margin(10, 10, 10, 10), # 调整图例的边距
    legend.spacing.y = unit(0.5,'cm')
  )  + scale_color_manual(values = c("#E64B35", "#4DBBD5", "#E6C02B",
                                     "#147A5C", "#8700FF", "#800000"),
                          name = 'AUC of each model'
  )
ROC_curves
save_path = paste0('F:/DKI/final_results/plot/Train Set_ROC.tiff')
ggsave(save_path, plot = ROC_curves,
       width = 3900,height = 3000,units = 'px',dpi = 300)

#############################time-ROC#############################
internal_test_Result <- read.csv(result_paths[1], header = TRUE, row.names = 1)
external_test_Result <- read.csv(result_paths[2], header = TRUE, row.names = 1)
results_names <- c('internal_test_Result', 'external_test_Result')
for (i in seq_along(predictors)){
  predictor <- predictors[i]
  model_name <- model_names[i]
  internal_time_roc_res <- timeROC(
    T = internal_test_Result$DFS_time,
    delta = internal_test_Result$CLS_GT,
    marker = internal_test_Result[[predictor]],
    cause = 1,
    weighting="marginal",
    times = c(1 *12, 3 * 12, 5 * 12),
    ROC = TRUE,
    iid = TRUE
  )
  confint(internal_time_roc_res,level = 0.95)$CI_AUC
  internal_time_roc_CI <- confint(internal_time_roc_res,level = 0.95)$CI_AUC/100
  for (i in 1:nrow(internal_time_roc_CI)) {
    for (j in 1:ncol(internal_time_roc_CI)) {
      # 判断当前值是否大于1
      if (internal_time_roc_CI[i, j] > 1) {
        # 将大于1的值设为1
        internal_time_roc_CI[i, j] <- 1
      }
    }
  }
  
  internal_time_ROC_df1 <- data.frame(
    TP = internal_time_roc_res$TP[, 1],
    FP = internal_time_roc_res$FP[, 1],
    time = c(rep(1, length(internal_time_roc_res$TP)))
  )
  internal_time_ROC_df2 <- data.frame(
    TP = internal_time_roc_res$TP[, 2],
    FP = internal_time_roc_res$FP[, 2],
    time = c(rep(2, length(internal_time_roc_res$TP)))
  )
  internal_time_ROC_df3 <- data.frame(
    TP = internal_time_roc_res$TP[, 3],
    FP = internal_time_roc_res$FP[, 3],
    time = c(rep(3, length(internal_time_roc_res$TP)))
  )
  internal_time_ROC_df <- rbind(internal_time_ROC_df1, internal_time_ROC_df2)
  internal_time_ROC_df <- rbind(internal_time_ROC_df, internal_time_ROC_df3)
  # internal_time_ROC_df$set <- 1
  internal_time_ROC_df$time <- factor(internal_time_ROC_df$time)
  
  ROC_curves_internal <- ggplot(data = internal_time_ROC_df,
                             aes(x = FP, y = TP, col = time))+
    geom_line(size = 1.5) +
    geom_abline(slope = 1, intercept = 0, color = "grey", size = 1, linetype = 2) +
    theme_bw() +
    labs(x = "False positive rate", y = "True positive rate") +
    theme(
      axis.text = element_text(size = 11, color = "black"),
      axis.title.x = element_text(face = "bold", size = 14, color = "black"),
      axis.title.y = element_text(face = "bold", size = 14, color = "black"),
      legend.position = c(0.7,0.15), legend.background = element_blank(),
      legend.box.background = element_rect(fill = NA, 
                                           colour = 'black', 
                                           linetype = 1, 
                                           size = 1.5),
      legend.text = element_text(size = 20),
      legend.title = element_text(size = 18, face = 'bold'),
      legend.margin = margin(10, 10, 10, 10), # 调整图例的边距
      legend.spacing.y = unit(0.5,'cm')
    )  + scale_color_manual(values = c("#E64B35", "#4DBBD5", "#E6C02B"),
                            name = 'Internal Testing Set',
                            labels = c(paste0("1 year AUC: ", 
                                              sprintf("%.2f (%.2f-%.2f)", 
                                                      internal_time_roc_res$AUC[[1]],
                                                      internal_time_roc_CI[1,1],
                                                      internal_time_roc_CI[1,2])),
                                       paste0("3 year AUC: ", 
                                              sprintf("%.2f (%.2f-%.2f)", 
                                                      internal_time_roc_res$AUC[[2]],
                                                      internal_time_roc_CI[2,1],
                                                      internal_time_roc_CI[2,2])),
                                       paste0("5 year AUC: ", 
                                              sprintf("%.2f (%.2f-%.2f)", 
                                                      internal_time_roc_res$AUC[[3]],
                                                      internal_time_roc_CI[3,1],
                                                      internal_time_roc_CI[3,2]))
                                       ))
  
  print(ROC_curves_internal)
  save_path = paste0('F:/DKI/final_results/plot/timeROC_', model_name, '_internal.tiff')
  ggsave(save_path, plot = ROC_curves_internal,
         width = 3900,height = 3000,units = 'px',dpi = 300)
}
for (i in seq_along(predictors)){
  predictor <- predictors[i]
  model_name <- model_names[i]
  external_time_roc_res <- timeROC(
    T = external_test_Result$DFS_time,
    delta = external_test_Result$CLS_GT,
    marker = external_test_Result[[predictor]],
    cause = 1,
    weighting="marginal",
    times = c(1 *12, 3 * 12, 5 * 12),
    ROC = TRUE,
    iid = TRUE
  )
  confint(external_time_roc_res,level = 0.95)$CI_AUC
  external_time_roc_CI <- confint(external_time_roc_res,level = 0.95)$CI_AUC/100
  for (i in 1:nrow(external_time_roc_CI)) {
    for (j in 1:ncol(external_time_roc_CI)) {
      # 判断当前值是否大于1
      if (external_time_roc_CI[i, j] > 1) {
        # 将大于1的值设为1
        external_time_roc_CI[i, j] <- 1
      }
    }
  }
  
  external_time_ROC_df1 <- data.frame(
    TP = external_time_roc_res$TP[, 1],
    FP = external_time_roc_res$FP[, 1],
    time = c(rep(1, length(external_time_roc_res$TP)))
  )
  external_time_ROC_df2 <- data.frame(
    TP = external_time_roc_res$TP[, 2],
    FP = external_time_roc_res$FP[, 2],
    time = c(rep(2, length(external_time_roc_res$TP)))
  )
  external_time_ROC_df3 <- data.frame(
    TP = external_time_roc_res$TP[, 3],
    FP = external_time_roc_res$FP[, 3],
    time = c(rep(3, length(external_time_roc_res$TP)))
  )
  external_time_ROC_df <- rbind(external_time_ROC_df1, external_time_ROC_df2)
  external_time_ROC_df <- rbind(external_time_ROC_df, external_time_ROC_df3)
  # external_time_ROC_df$set <- 2
  external_time_ROC_df$time <- factor(external_time_ROC_df$time)
  
  ROC_curves_external <- ggplot(data = external_time_ROC_df,
                                aes(x = FP, y = TP, col = time))+
    geom_line(size = 1.5) +
    geom_abline(slope = 1, intercept = 0, color = "grey", size = 1, linetype = 2) +
    theme_bw() +
    labs(x = "False positive rate", y = "True positive rate") +
    theme(
      axis.text = element_text(size = 11, color = "black"),
      axis.title.x = element_text(face = "bold", size = 14, color = "black"),
      axis.title.y = element_text(face = "bold", size = 14, color = "black"),
      legend.position = c(0.7,0.15), legend.background = element_blank(),
      legend.box.background = element_rect(fill = NA, 
                                           colour = 'black', 
                                           linetype = 1, 
                                           size = 1.5),
      legend.text = element_text(size = 20),
      legend.title = element_text(size = 18, face = 'bold'),
      legend.margin = margin(10, 10, 10, 10), # 调整图例的边距
      legend.spacing.y = unit(0.5,'cm')
    )  + scale_color_manual(values = c("#E64B35", "#4DBBD5", "#E6C02B"),
                            name = 'External Testing Set',
                            labels = c(paste0("1 year AUC: NA"),
                                       paste0("3 year AUC: ", 
                                              sprintf("%.2f (%.2f-%.2f)", 
                                                      external_time_roc_res$AUC[[2]],
                                                      external_time_roc_CI[1,1],
                                                      external_time_roc_CI[1,2])),
                                       paste0("5 year AUC: ", 
                                              sprintf("%.2f (%.2f-%.2f)", 
                                                      external_time_roc_res$AUC[[3]],
                                                      external_time_roc_CI[2,1],
                                                      external_time_roc_CI[2,2]))
                            ))
  
  print(ROC_curves_external)
  save_path = paste0('F:/DKI/final_results/plot/timeROC_', model_name, '_external.tiff')
  ggsave(save_path, plot = ROC_curves_external,
         width = 3900,height = 3000,units = 'px',dpi = 300)
}
#########################DCA########################################
library(dcurves)
library(survival)
data = internal_test_Result
data = external_test_Result
dcurves::dca(Surv(DFS_time,CLS_GT) ~ all_risk+cli_risk+precli_risk+
               t2_risk+img_risk+img_precli_risk,
             data = data,
             time = 60) %>%
  plot(smooth = TRUE,show_ggplot_code = T) +
  ggplot2::theme_bw() +
  labs(x = "Treatment Threshold Probability") +
  theme(
    axis.text = element_text(size = 20, color = "black"),
    axis.title.x = element_text(face = "bold", size = 20, color = "black", 
                                margin = margin(c(15, 0, 0, 0))),
    axis.title.y = element_text(face = "bold", size = 20, color = "black", 
                                margin = margin(c(0, 15, 0, 0))),
    legend.position = c(0.85,0.9), legend.background = element_blank(),
    legend.text = element_text(size = 20),
    legend.title = element_text(size = 18, face = 'bold'),
    legend.margin = margin(0, 0, 0, 0)
  )+
  scale_color_manual(values = c('darkgrey','grey',
                                "#E64B35", "#4DBBD5", "#E6C02B",
                                "#8700FF", "#147A5C", "#800000"),
                     name = '',
                     labels = c("All","None",
                                "All Model", 
                                "Clinical Model",
                                "Preclinical Model",
                                "T2 Model", 
                                "T2+DKI Model", 
                                "T2+DKI+Preclinical Model"))
ggsave('F:/DKI/final_results/plot/DCA Curve_external.tiff',width = 3900,
       height = 3000,units = 'px',dpi = 300)
  
#########################C-index####################################

library(survival)
library(pec)

predictors <- c('all_risk', 'cli_risk', 'precli_risk',
                't2_risk', 'img_risk', 'img_precli_risk')
# 创建空数据框存储结果
Cindex_results <- data.frame(Model = character(),
                             Predictor = character(),
                      C_Index = numeric(),
                      Std_Error = numeric(),
                      Lower_CI = numeric(),
                      Upper_CI = numeric(),
                      set = factor(),
                      stringsAsFactors = FALSE)


# 循环计算每个预测因子的C-index、标准误和95%置信区间
for (i in seq_along(results_names)) {  
  data <- get(results_names[i])
  for (j in seq_along(predictors)) {
    predictor <- predictors[j]
    model_name <- model_names[j]
    # 创建模型
    data$DFS_time <- as.numeric(data$DFS_time)
    data$CLS_GT <- as.numeric(data$CLS_GT)
    survData <- Surv(data$DFS_time,data$CLS_GT)
    formula <- as.formula(paste("survData ~ ", predictor))
    model <- coxph(formula, data = data)
    
    # 计算C-index
    sum.surv <- summary(model)
    cindex <- (sum.surv$concordance)[1]
    
    # 提取标准误和95%置信区间
    se <- (sum.surv$concordance)[2]
    lowerCI <- cindex-1.96*se
    upperCI <- cindex+1.96*se
    
    # 存储结果
    Cindex_results <- rbind(Cindex_results, data.frame(Model = model_name,
                                                       Predictor = predictor,
                                         C_Index = cindex,
                                         Std_Error = se,
                                         Lower_CI = lowerCI,
                                         Upper_CI = upperCI,
                                         set = i,
                                         stringsAsFactors = FALSE))}
  }
Cindex_results$set <- factor(Cindex_results$set, levels = c(1,2),
                             labels = cohort_names)
# 绘制箱型图
cindex_curves <- ggplot(Cindex_results, aes(x = set, y = C_Index, 
                    color = Predictor)) +
  geom_pointrange(aes(y = C_Index, ymin = Lower_CI, ymax = Upper_CI),
                  position = position_dodge(0.5), size = 1, fatten = 10) +
  geom_linerange(aes(y = C_Index, ymin = Lower_CI, ymax = Upper_CI),
                 position = position_dodge(0.5), size = 2) +
  theme_bw() +
  labs(x = "", y = "C-index") +
  theme(
    axis.text = element_text(size = 20, color = "black"),
    axis.title.x = element_text(face = "bold", size = 20, color = "black", 
                                margin = margin(c(15, 0, 0, 0))),
    axis.title.y = element_text(face = "bold", size = 20, color = "black", 
                                margin = margin(c(0, 15, 0, 0))),
    legend.position = 'bottom', legend.background = element_blank(),
    legend.text = element_text(size = 20),
    legend.title = element_text(size = 18, face = 'bold'),
    legend.margin = margin(0, 0, 0, 0), # 调整图例的边距
    legend.box = "horizontal" # 图例水平排列
  )  + scale_color_manual(values = c("#E64B35", "#4DBBD5", "#E6C02B",
                                     "#8700FF", "#147A5C", "#800000"),
                          name = '',
                          breaks = c('all_risk','cli_risk',
                                     'precli_risk',
                                     'img_risk', 't2_risk',
                                     'img_precli_risk' 
                                     ),
                          labels = c("All Model", 
                            "Clinical Model",
                            "Preclinical Model",
                            "T2+DKI Model", 
                            "T2 Model", 
                            "T2+DKI+Preclinical Model"))+
  guides(color = guide_legend(ncol = 6)) # 设置足够大的ncol值
cindex_curves
ggsave('F:/DKI/final_results/plot/C-index.tiff', plot = cindex_curves,
       width = 4500,height = 3000,units = 'px',dpi = 300)

########################KM#######################
library(survival)
library(survminer)
library(cowplot)
library(dplyr)
for (i in seq_along(predictors)) {
  predictor <- predictors[i]
  model_name <- model_names[i]
  internal_test_Result$risk_level <- ifelse(internal_test_Result[predictor]>0.5,
                                            1, 0)
  internal_test_Result$risk_level <- factor(internal_test_Result$risk_level, 
                                            levels = c(0,1),
                                            labels = c('Low risk', 'High risk'))
  DFS_survdiff <- survdiff(Surv(DFS_time,CLS_GT)~risk_level, 
                           data = internal_test_Result)
  DFS_pval = 1 - pchisq(DFS_survdiff$chisq, length(DFS_survdiff$n) - 1)
  DFS_HR = (DFS_survdiff$obs[2]/DFS_survdiff$exp[2])/(DFS_survdiff$obs[1]/DFS_survdiff$exp[1])
  DFS_up95 = exp(log(DFS_HR) + qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1]))
  DFS_low95 = exp(log(DFS_HR) - qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1])) 
  DFS_fit <- survfit(Surv(DFS_time,CLS_GT)~risk_level, data = internal_test_Result)
  DFS_P <- ggsurvplot(DFS_fit,
                      title = paste0("K-M curve of ", model_name,
                                     " in internal testing set"),
                      size = 1.7,
                      censor.size = 9,
                      conf.int = T,# ??ʾ????????
                      pval = F,
                      risk.table = TRUE,
                      fontsize = 6,
                      break.x.by = 12,
                      linetype = 1,# ?????Ա??????Զ?????????????
                      surv.median.line = "hv", # ??????λ????????ʾ
                      ggtheme = theme_bw()+theme(
                        title = element_text(face = "bold", 
                                             size = 20, color = "black"), 
                        axis.text = element_text(face = "bold", 
                                                 size = 16, color = "black"),
                        axis.title.x = element_text(face = "bold", size = 15, 
                                                    color = "black" ),
                        axis.title.y = element_text(face = "bold", size = 15, 
                                                    color = "black" ), 
                        panel.grid.minor = element_blank(),
                        panel.grid.major = element_blank(),
                        legend.text = element_text(size = 15)), 
                      palette = c("#2F4858", "#B01414"),
                      xlab = 'DFS Time (month)',
                      legend.title = '',
                      legend.labs = c('Low risk', 'High risk'))
  p_val <- ifelse(DFS_pval<0.001, '<0.001', sprintf("%.3f", DFS_pval))
  HR = paste0("HR = ", 
                 sprintf("%.2f (%.2f-%.2f)", 
                         DFS_HR,
                         DFS_low95,
                         DFS_up95))
  
  print(predictor)
  print("Internal Testing Set")
  print(p_val)
  print(HR)
  print(DFS_P)
  combined_plot <- plot_grid(DFS_P$plot, DFS_P$table,
                             nrow = 2, align = "v", axis = "lr",
                             rel_heights = c(5,1))
  save_path = paste0('F:/DKI/final_results/plot/new_KM/internal_', model_name, '_KM.tiff')
  ggsave(save_path, plot = combined_plot, device =  "tiff",
         width = 3900,height = 3000,units = 'px',dpi = 300)

  external_test_Result$risk_level <- ifelse(external_test_Result[predictor]>0.5,
                                            1, 0)
  external_test_Result$risk_level <- factor(external_test_Result$risk_level,
                                            levels = c(0,1),
                                            labels = c('low risk', 'high risk'))
  DFS_survdiff <- survdiff(Surv(DFS_time,CLS_GT)~risk_level,
                           data = external_test_Result)
  DFS_pval = 1 - pchisq(DFS_survdiff$chisq, length(DFS_survdiff$n) - 1)
  DFS_HR = (DFS_survdiff$obs[2]/DFS_survdiff$exp[2])/(DFS_survdiff$obs[1]/DFS_survdiff$exp[1])
  DFS_up95 = exp(log(DFS_HR) + qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1]))
  DFS_low95 = exp(log(DFS_HR) - qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1]))
  DFS_fit <- survfit(Surv(DFS_time,CLS_GT)~risk_level, data = external_test_Result)
  DFS_P <- ggsurvplot(DFS_fit,
                      title = paste0("K-M curve of ", model_name,
                                     " in external testing set"),
                      size = 1.7,
                      censor.size = 9,
                      conf.int = T,# ??ʾ????????
                      pval = F,
                      risk.table = TRUE,
                      fontsize = 6,
                      break.x.by = 12,
                      linetype = 1,# ?????Ա??????Զ?????????????
                      surv.median.line = "hv", # ??????λ????????ʾ
                      ggtheme = theme_bw()+theme(
                        title = element_text(face = "bold", 
                                             size = 20, color = "black"), 
                        axis.text = element_text(face = "bold", 
                                                 size = 16, color = "black"),
                        axis.title.x = element_text(face = "bold", size = 15, 
                                                    color = "black" ),
                        axis.title.y = element_text(face = "bold", size = 15, 
                                                    color = "black" ), 
                        panel.grid.minor = element_blank(),
                        panel.grid.major = element_blank(),
                        legend.text = element_text(size = 15)), 
                      palette = c("#2F4858", "#B01414"),
                      xlab = 'DFS Time (month)',
                      legend.title = '',
                      legend.labs = c('Low risk', 'High risk'))
  p_val <- ifelse(DFS_pval<0.001, '<0.001', sprintf("%.3f", DFS_pval))
  HR = paste0("HR = ", 
              sprintf("%.2f (%.2f-%.2f)", 
                      DFS_HR,
                      DFS_low95,
                      DFS_up95))
  print('----------------------------')
  print("External Testing Set")
  print(p_val)
  print(HR)
  print('#############################')
  print(DFS_P)
  combined_plot <- plot_grid(DFS_P$plot, DFS_P$table,
                             nrow = 2, align = "v", axis = "lr",
                             rel_heights = c(5,1))
  save_path = paste0('F:/DKI/final_results/plot/new_KM/external_', model_name, '_KM.tiff')
  ggsave(save_path, plot = combined_plot, device =  "tiff",
         width = 3900,height = 3000,units = 'px',dpi = 300)
}

predictors <-  c('all_risk',
                 'img_precli_risk' )
model_names <- c("All Model", 
                 "T2+DKI+Preclinical Model")
for (i in seq_along(cohort_names)){
  cohort_name <- cohort_names[i]
  print(cohort_name)
  result_data <- get(results_names[i])
  sub_clinical_data <- clinical_data %>% 
    filter(Cohort == cohort_name)
  row.names(sub_clinical_data) <- sub_clinical_data$id_name
  matched_df <- merge(sub_clinical_data, result_data, by = "row.names", all = TRUE)
  matched_df <- matched_df[, -1]
  subgroup_1 <- matched_df %>% 
    filter(TRG > 1)
  subgroup_0 <- matched_df %>% 
    filter(TRG < 2)
  for (i in seq_along(predictors)) {
    predictor <- predictors[i]
    model_name <- model_names[i]
    subgroup_1$risk_level <- ifelse(subgroup_1[predictor]>0.5,
                                              1, 0)
    subgroup_1$risk_level <- factor(subgroup_1$risk_level, 
                                              levels = c(0,1),
                                              labels = c('Low risk', 'High risk'))
    DFS_survdiff <- survdiff(Surv(DFS_time,CLS_GT)~risk_level, 
                             data = subgroup_1)
    DFS_pval = 1 - pchisq(DFS_survdiff$chisq, length(DFS_survdiff$n) - 1)
    DFS_HR = (DFS_survdiff$obs[2]/DFS_survdiff$exp[2])/(DFS_survdiff$obs[1]/DFS_survdiff$exp[1])
    DFS_up95 = exp(log(DFS_HR) + qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1]))
    DFS_low95 = exp(log(DFS_HR) - qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1])) 
    DFS_fit <- survfit(Surv(DFS_time,CLS_GT)~risk_level, data = subgroup_1)
    DFS_P <- ggsurvplot(DFS_fit,
                        title = paste0("K-M curve of ", model_name,
                                       " for poor responders based on ", cohort_name),
                        size = 1.7,
                        censor.size = 9,
                        conf.int = T,# ??ʾ????????
                        pval = F,
                        risk.table = TRUE,
                        fontsize = 6,
                        break.x.by = 12,
                        linetype = 1,# ?????Ա??????Զ?????????????
                        surv.median.line = "hv", # ??????λ????????ʾ
                        ggtheme = theme_bw()+theme(
                          title = element_text(face = "bold", 
                                               size = 15, color = "black"), 
                          axis.text = element_text(face = "bold", 
                                                   size = 16, color = "black"),
                          axis.title.x = element_text(face = "bold", size = 15, 
                                                      color = "black" ),
                          axis.title.y = element_text(face = "bold", size = 15, 
                                                      color = "black" ), 
                          panel.grid.minor = element_blank(),
                          panel.grid.major = element_blank(),
                          legend.text = element_text(size = 15)), 
                        palette = c("#2F4858", "#B01414"),
                        xlab = 'DFS Time (month)',
                        legend.title = '',
                        legend.labs = c('Low risk', 'High risk'))
    p_val <- ifelse(DFS_pval<0.001, '<0.001', sprintf("%.3f", DFS_pval))
    HR = paste0("HR = ", 
                sprintf("%.2f (%.2f-%.2f)", 
                        DFS_HR,
                        DFS_low95,
                        DFS_up95))
    
    print(predictor)
    print("poor responders")
    print(p_val)
    print(HR)
    print(DFS_P)
    combined_plot <- plot_grid(DFS_P$plot, DFS_P$table,
                               nrow = 2, align = "v", axis = "lr",
                               rel_heights = c(5,1))
    save_path = paste0('F:/DKI/final_results/plot/new_KM/', 
                       cohort_name, '_', model_name, '_KM_TRG1.tiff')
    ggsave(save_path, plot = combined_plot, device =  "tiff",
           width = 3900,height = 3000,units = 'px',dpi = 300)
    
    subgroup_0$risk_level <- ifelse(subgroup_0[predictor]>0.5,
                                              1, 0)
    subgroup_0$risk_level <- factor(subgroup_0$risk_level, 
                                              levels = c(0,1),
                                              labels = c('low risk', 'high risk'))
    DFS_survdiff <- survdiff(Surv(DFS_time,CLS_GT)~risk_level, 
                             data = subgroup_0)
    DFS_pval = 1 - pchisq(DFS_survdiff$chisq, length(DFS_survdiff$n) - 1)
    DFS_HR = (DFS_survdiff$obs[2]/DFS_survdiff$exp[2])/(DFS_survdiff$obs[1]/DFS_survdiff$exp[1])
    DFS_up95 = exp(log(DFS_HR) + qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1]))
    DFS_low95 = exp(log(DFS_HR) - qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1])) 
    DFS_fit <- survfit(Surv(DFS_time,CLS_GT)~risk_level, data = subgroup_0)
    DFS_P <- ggsurvplot(DFS_fit,
                        title = paste0("K-M curve of ", model_name,
                                       " for good responders based on ", cohort_name),
                        size = 1.7,
                        censor.size = 9,
                        conf.int = T,# ??ʾ????????
                        pval = F,
                        risk.table = TRUE,
                        fontsize = 6,
                        break.x.by = 12,
                        linetype = 1,# ?????Ա??????Զ?????????????
                        surv.median.line = "hv", # ??????λ????????ʾ
                        ggtheme = theme_bw()+theme(
                          title = element_text(face = "bold", 
                                               size = 15, color = "black"), 
                          axis.text = element_text(face = "bold", 
                                                   size = 16, color = "black"),
                          axis.title.x = element_text(face = "bold", size = 15, 
                                                      color = "black" ),
                          axis.title.y = element_text(face = "bold", size = 15, 
                                                      color = "black" ), 
                          panel.grid.minor = element_blank(),
                          panel.grid.major = element_blank(),
                          legend.text = element_text(size = 15)), 
                        palette = c("#2F4858", "#B01414"),
                        xlab = 'DFS Time (month)',
                        legend.title = '',
                        legend.labs = c('Low risk', 'High risk'))
    p_val <- ifelse(DFS_pval<0.001, '<0.001', sprintf("%.3f", DFS_pval))
    HR = paste0("HR = ", 
                sprintf("%.2f (%.2f-%.2f)", 
                        DFS_HR,
                        DFS_low95,
                        DFS_up95))
    print('----------------------------')
    print("good responders")
    print(p_val)
    print(HR)
    print('#############################')
    print(DFS_P)
    combined_plot <- plot_grid(DFS_P$plot, DFS_P$table,
                               nrow = 2, align = "v", axis = "lr",
                               rel_heights = c(5,1))
    save_path = paste0('F:/DKI/final_results/plot/new_KM/', 
                       cohort_name, '_', model_name, '_KM_TRG0.tiff')
    ggsave(save_path, plot = combined_plot, device =  "tiff",
           width = 3900,height = 3000,units = 'px',dpi = 300)
  }
}

cohort_name <- cohort_names[2]
print(cohort_name)
result_data <- get(results_names[2])
sub_clinical_data <- clinical_data %>% 
  filter(Cohort == cohort_name)
row.names(sub_clinical_data) <- sub_clinical_data$id_name
matched_df <- merge(sub_clinical_data, result_data, by = "row.names", all = TRUE)
matched_df <- matched_df[, -1]

subgroup_1 <- matched_df %>% 
  filter(Location == '<5cm')
subgroup_0 <- matched_df %>% 
  filter(Location != '<5cm')
for (i in seq_along(predictors)) {
  predictor <- predictors[i]
  model_name <- model_names[i]
  subgroup_1$risk_level <- ifelse(subgroup_1[predictor]>0.5,
                                  1, 0)
  subgroup_1$risk_level <- factor(subgroup_1$risk_level, 
                                  levels = c(0,1),
                                  labels = c('Low risk', 'High risk'))
  DFS_survdiff <- survdiff(Surv(DFS_time,CLS_GT)~risk_level, 
                           data = subgroup_1)
  DFS_pval = 1 - pchisq(DFS_survdiff$chisq, length(DFS_survdiff$n) - 1)
  DFS_HR = (DFS_survdiff$obs[2]/DFS_survdiff$exp[2])/(DFS_survdiff$obs[1]/DFS_survdiff$exp[1])
  DFS_up95 = exp(log(DFS_HR) + qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1]))
  DFS_low95 = exp(log(DFS_HR) - qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1])) 
  DFS_fit <- survfit(Surv(DFS_time,CLS_GT)~risk_level, data = subgroup_1)
  DFS_P <- ggsurvplot(DFS_fit,
                      title = paste0("K-M curve of ", model_name,
                                     " for patients with low tumor locations"),
                      size = 1.7,
                      censor.size = 9,
                      conf.int = T,# ??ʾ????????
                      pval = F,
                      risk.table = TRUE,
                      fontsize = 6,
                      break.x.by = 12,
                      linetype = 1,# ?????Ա??????Զ?????????????
                      surv.median.line = "hv", # ??????λ????????ʾ
                      ggtheme = theme_bw()+theme(
                        title = element_text(face = "bold", 
                                             size = 15, color = "black"), 
                        axis.text = element_text(face = "bold", 
                                                 size = 16, color = "black"),
                        axis.title.x = element_text(face = "bold", size = 15, 
                                                    color = "black" ),
                        axis.title.y = element_text(face = "bold", size = 15, 
                                                    color = "black" ), 
                        panel.grid.minor = element_blank(),
                        panel.grid.major = element_blank(),
                        legend.text = element_text(size = 15)), 
                      palette = c("#2F4858", "#B01414"),
                      xlab = 'DFS Time (month)',
                      legend.title = '',
                      legend.labs = c('Low risk', 'High risk'))
  p_val <- ifelse(DFS_pval<0.001, '<0.001', sprintf("%.3f", DFS_pval))
  HR = paste0("HR = ", 
              sprintf("%.2f (%.2f-%.2f)", 
                      DFS_HR,
                      DFS_low95,
                      DFS_up95))
  
  print(predictor)
  print("low")
  print(p_val)
  print(HR)
  print(DFS_P)
  combined_plot <- plot_grid(DFS_P$plot, DFS_P$table,
                             nrow = 2, align = "v", axis = "lr",
                             rel_heights = c(5,1))
  save_path = paste0('F:/DKI/final_results/plot/new_KM/', 
                     cohort_name, '_', model_name, '_KM_low.tiff')
  ggsave(save_path, plot = combined_plot, device =  "tiff",
         width = 3900,height = 3000,units = 'px',dpi = 300)
  
  subgroup_0$risk_level <- ifelse(subgroup_0[predictor]>0.5,
                                  1, 0)
  subgroup_0$risk_level <- factor(subgroup_0$risk_level, 
                                  levels = c(0,1),
                                  labels = c('low risk', 'high risk'))
  DFS_survdiff <- survdiff(Surv(DFS_time,CLS_GT)~risk_level, 
                           data = subgroup_0)
  DFS_pval = 1 - pchisq(DFS_survdiff$chisq, length(DFS_survdiff$n) - 1)
  DFS_HR = (DFS_survdiff$obs[2]/DFS_survdiff$exp[2])/(DFS_survdiff$obs[1]/DFS_survdiff$exp[1])
  DFS_up95 = exp(log(DFS_HR) + qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1]))
  DFS_low95 = exp(log(DFS_HR) - qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1])) 
  DFS_fit <- survfit(Surv(DFS_time,CLS_GT)~risk_level, data = subgroup_0)
  DFS_P <- ggsurvplot(DFS_fit,
                      title = paste0("K-M curve of ", model_name,
                                     " for patients with medium-high tumor locations"),
                      size = 1.7,
                      censor.size = 9,
                      conf.int = T,# ??ʾ????????
                      pval = F,
                      risk.table = TRUE,
                      fontsize = 6,
                      break.x.by = 12,
                      linetype = 1,# ?????Ա??????Զ?????????????
                      surv.median.line = "hv", # ??????λ????????ʾ
                      ggtheme = theme_bw()+theme(
                        title = element_text(face = "bold", 
                                             size = 15, color = "black"), 
                        axis.text = element_text(face = "bold", 
                                                 size = 16, color = "black"),
                        axis.title.x = element_text(face = "bold", size = 15, 
                                                    color = "black" ),
                        axis.title.y = element_text(face = "bold", size = 15, 
                                                    color = "black" ), 
                        panel.grid.minor = element_blank(),
                        panel.grid.major = element_blank(),
                        legend.text = element_text(size = 15)), 
                      palette = c("#2F4858", "#B01414"),
                      xlab = 'DFS Time (month)',
                      legend.title = '',
                      legend.labs = c('Low risk', 'High risk'))
  p_val <- ifelse(DFS_pval<0.001, '<0.001', sprintf("%.3f", DFS_pval))
  HR = paste0("HR = ", 
              sprintf("%.2f (%.2f-%.2f)", 
                      DFS_HR,
                      DFS_low95,
                      DFS_up95))
  print('----------------------------')
  print("medium-high")
  print(p_val)
  print(HR)
  print('#############################')
  print(DFS_P)
  combined_plot <- plot_grid(DFS_P$plot, DFS_P$table,
                             nrow = 2, align = "v", axis = "lr",
                             rel_heights = c(5,1))
  save_path = paste0('F:/DKI/final_results/plot/new_KM/', 
                     cohort_name, '_', model_name, '_KM_mid2high.tiff')
  ggsave(save_path, plot = combined_plot, device =  "tiff",
         width = 3900,height = 3000,units = 'px',dpi = 300)
}

subgroup_1 <- matched_df %>% 
  filter(EMVI == '+')
subgroup_0 <- matched_df %>% 
  filter(EMVI == '-')
for (i in seq_along(predictors)) {
  predictor <- predictors[i]
  model_name <- model_names[i]
  subgroup_1$risk_level <- ifelse(subgroup_1[predictor]>0.5,
                                  1, 0)
  subgroup_1$risk_level <- factor(subgroup_1$risk_level, 
                                  levels = c(0,1),
                                  labels = c('Low risk', 'High risk'))
  DFS_survdiff <- survdiff(Surv(DFS_time,CLS_GT)~risk_level, 
                           data = subgroup_1)
  DFS_pval = 1 - pchisq(DFS_survdiff$chisq, length(DFS_survdiff$n) - 1)
  DFS_HR = (DFS_survdiff$obs[2]/DFS_survdiff$exp[2])/(DFS_survdiff$obs[1]/DFS_survdiff$exp[1])
  DFS_up95 = exp(log(DFS_HR) + qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1]))
  DFS_low95 = exp(log(DFS_HR) - qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1])) 
  DFS_fit <- survfit(Surv(DFS_time,CLS_GT)~risk_level, data = subgroup_1)
  DFS_P <- ggsurvplot(DFS_fit,
                      title = paste0("K-M curve of ", model_name,
                                     " for patients with EMVI"),
                      size = 1.7,
                      censor.size = 9,
                      conf.int = T,# ??ʾ????????
                      pval = F,
                      risk.table = TRUE,
                      fontsize = 6,
                      break.x.by = 12,
                      linetype = 1,# ?????Ա??????Զ?????????????
                      surv.median.line = "hv", # ??????λ????????ʾ
                      ggtheme = theme_bw()+theme(
                        title = element_text(face = "bold", 
                                             size = 15, color = "black"), 
                        axis.text = element_text(face = "bold", 
                                                 size = 16, color = "black"),
                        axis.title.x = element_text(face = "bold", size = 15, 
                                                    color = "black" ),
                        axis.title.y = element_text(face = "bold", size = 15, 
                                                    color = "black" ), 
                        panel.grid.minor = element_blank(),
                        panel.grid.major = element_blank(),
                        legend.text = element_text(size = 15)), 
                      palette = c("#2F4858", "#B01414"),
                      xlab = 'DFS Time (month)',
                      legend.title = '',
                      legend.labs = c('Low risk', 'High risk'))
  p_val <- ifelse(DFS_pval<0.001, '<0.001', sprintf("%.3f", DFS_pval))
  HR = paste0("HR = ", 
              sprintf("%.2f (%.2f-%.2f)", 
                      DFS_HR,
                      DFS_low95,
                      DFS_up95))
  
  print(predictor)
  print("EMVI1")
  print(p_val)
  print(HR)
  print(DFS_P)
  combined_plot <- plot_grid(DFS_P$plot, DFS_P$table,
                             nrow = 2, align = "v", axis = "lr",
                             rel_heights = c(5,1))
  save_path = paste0('F:/DKI/final_results/plot/new_KM/', 
                     cohort_name, '_', model_name, '_KM_EMVI1.tiff')
  ggsave(save_path, plot = combined_plot, device =  "tiff",
         width = 3900,height = 3000,units = 'px',dpi = 300)
  
  subgroup_0$risk_level <- ifelse(subgroup_0[predictor]>0.5,
                                  1, 0)
  subgroup_0$risk_level <- factor(subgroup_0$risk_level, 
                                  levels = c(0,1),
                                  labels = c('low risk', 'high risk'))
  DFS_survdiff <- survdiff(Surv(DFS_time,CLS_GT)~risk_level, 
                           data = subgroup_0)
  DFS_pval = 1 - pchisq(DFS_survdiff$chisq, length(DFS_survdiff$n) - 1)
  DFS_HR = (DFS_survdiff$obs[2]/DFS_survdiff$exp[2])/(DFS_survdiff$obs[1]/DFS_survdiff$exp[1])
  DFS_up95 = exp(log(DFS_HR) + qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1]))
  DFS_low95 = exp(log(DFS_HR) - qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1])) 
  DFS_fit <- survfit(Surv(DFS_time,CLS_GT)~risk_level, data = subgroup_0)
  DFS_P <- ggsurvplot(DFS_fit,
                      title = paste0("K-M curve of ", model_name,
                                     " for patients without EMVI"),
                      size = 1.7,
                      censor.size = 9,
                      conf.int = T,# ??ʾ????????
                      pval = F,
                      risk.table = TRUE,
                      fontsize = 6,
                      break.x.by = 12,
                      linetype = 1,# ?????Ա??????Զ?????????????
                      surv.median.line = "hv", # ??????λ????????ʾ
                      ggtheme = theme_bw()+theme(
                        title = element_text(face = "bold", 
                                             size = 15, color = "black"), 
                        axis.text = element_text(face = "bold", 
                                                 size = 16, color = "black"),
                        axis.title.x = element_text(face = "bold", size = 15, 
                                                    color = "black" ),
                        axis.title.y = element_text(face = "bold", size = 15, 
                                                    color = "black" ), 
                        panel.grid.minor = element_blank(),
                        panel.grid.major = element_blank(),
                        legend.text = element_text(size = 15)), 
                      palette = c("#2F4858", "#B01414"),
                      xlab = 'DFS Time (month)',
                      legend.title = '',
                      legend.labs = c('Low risk', 'High risk'))
  p_val <- ifelse(DFS_pval<0.001, '<0.001', sprintf("%.3f", DFS_pval))
  HR = paste0("HR = ", 
              sprintf("%.2f (%.2f-%.2f)", 
                      DFS_HR,
                      DFS_low95,
                      DFS_up95))
  print('----------------------------')
  print("EMVI0")
  print(p_val)
  print(HR)
  print('#############################')
  print(DFS_P)
  combined_plot <- plot_grid(DFS_P$plot, DFS_P$table,
                             nrow = 2, align = "v", axis = "lr",
                             rel_heights = c(5,1))
  save_path = paste0('F:/DKI/final_results/plot/new_KM/', 
                     cohort_name, '_', model_name, '_KM_EMVI0.tiff')
  ggsave(save_path, plot = combined_plot, device =  "tiff",
         width = 3900,height = 3000,units = 'px',dpi = 300)
}

subgroup_1 <- matched_df %>% 
  filter(AC == '+')
subgroup_0 <- matched_df %>% 
  filter(AC == '-')
for (i in seq_along(predictors)) {
  predictor <- predictors[i]
  model_name <- model_names[i]
  subgroup_1$risk_level <- ifelse(subgroup_1[predictor]>0.5,
                                  1, 0)
  subgroup_1$risk_level <- factor(subgroup_1$risk_level, 
                                  levels = c(0,1),
                                  labels = c('Low risk', 'High risk'))
  DFS_survdiff <- survdiff(Surv(DFS_time,CLS_GT)~risk_level, 
                           data = subgroup_1)
  DFS_pval = 1 - pchisq(DFS_survdiff$chisq, length(DFS_survdiff$n) - 1)
  DFS_HR = (DFS_survdiff$obs[2]/DFS_survdiff$exp[2])/(DFS_survdiff$obs[1]/DFS_survdiff$exp[1])
  DFS_up95 = exp(log(DFS_HR) + qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1]))
  DFS_low95 = exp(log(DFS_HR) - qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1])) 
  DFS_fit <- survfit(Surv(DFS_time,CLS_GT)~risk_level, data = subgroup_1)
  DFS_P <- ggsurvplot(DFS_fit,
                      title = paste0("K-M curve of ", model_name,
                                     " for patients with AC"),
                      size = 1.7,
                      censor.size = 9,
                      conf.int = T,# ??ʾ????????
                      pval = F,
                      risk.table = TRUE,
                      fontsize = 6,
                      break.x.by = 12,
                      linetype = 1,# ?????Ա??????Զ?????????????
                      surv.median.line = "hv", # ??????λ????????ʾ
                      ggtheme = theme_bw()+theme(
                        title = element_text(face = "bold", 
                                             size = 15, color = "black"), 
                        axis.text = element_text(face = "bold", 
                                                 size = 16, color = "black"),
                        axis.title.x = element_text(face = "bold", size = 15, 
                                                    color = "black" ),
                        axis.title.y = element_text(face = "bold", size = 15, 
                                                    color = "black" ), 
                        panel.grid.minor = element_blank(),
                        panel.grid.major = element_blank(),
                        legend.text = element_text(size = 15)), 
                      palette = c("#2F4858", "#B01414"),
                      xlab = 'DFS Time (month)',
                      legend.title = '',
                      legend.labs = c('Low risk', 'High risk'))
  p_val <- ifelse(DFS_pval<0.001, '<0.001', sprintf("%.3f", DFS_pval))
  HR = paste0("HR = ", 
              sprintf("%.2f (%.2f-%.2f)", 
                      DFS_HR,
                      DFS_low95,
                      DFS_up95))
  
  print(predictor)
  print("AC1")
  print(p_val)
  print(HR)
  print(DFS_P)
  combined_plot <- plot_grid(DFS_P$plot, DFS_P$table,
                             nrow = 2, align = "v", axis = "lr",
                             rel_heights = c(5,1))
  save_path = paste0('F:/DKI/final_results/plot/new_KM/', 
                     cohort_name, '_', model_name, '_KM_AC1.tiff')
  ggsave(save_path, plot = combined_plot, device =  "tiff",
         width = 3900,height = 3000,units = 'px',dpi = 300)
  
  subgroup_0$risk_level <- ifelse(subgroup_0[predictor]>0.5,
                                  1, 0)
  subgroup_0$risk_level <- factor(subgroup_0$risk_level, 
                                  levels = c(0,1),
                                  labels = c('low risk', 'high risk'))
  DFS_survdiff <- survdiff(Surv(DFS_time,CLS_GT)~risk_level, 
                           data = subgroup_0)
  DFS_pval = 1 - pchisq(DFS_survdiff$chisq, length(DFS_survdiff$n) - 1)
  DFS_HR = (DFS_survdiff$obs[2]/DFS_survdiff$exp[2])/(DFS_survdiff$obs[1]/DFS_survdiff$exp[1])
  DFS_up95 = exp(log(DFS_HR) + qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1]))
  DFS_low95 = exp(log(DFS_HR) - qnorm(0.975)*sqrt(1/DFS_survdiff$exp[2]+1/DFS_survdiff$exp[1])) 
  DFS_fit <- survfit(Surv(DFS_time,CLS_GT)~risk_level, data = subgroup_0)
  DFS_P <- ggsurvplot(DFS_fit,
                      title = paste0("K-M curve of ", model_name,
                                     " for patients without AC"),
                      size = 1.7,
                      censor.size = 9,
                      conf.int = T,# ??ʾ????????
                      pval = F,
                      risk.table = TRUE,
                      fontsize = 6,
                      break.x.by = 12,
                      linetype = 1,# ?????Ա??????Զ?????????????
                      surv.median.line = "hv", # ??????λ????????ʾ
                      ggtheme = theme_bw()+theme(
                        title = element_text(face = "bold", 
                                             size = 15, color = "black"), 
                        axis.text = element_text(face = "bold", 
                                                 size = 16, color = "black"),
                        axis.title.x = element_text(face = "bold", size = 15, 
                                                    color = "black" ),
                        axis.title.y = element_text(face = "bold", size = 15, 
                                                    color = "black" ), 
                        panel.grid.minor = element_blank(),
                        panel.grid.major = element_blank(),
                        legend.text = element_text(size = 15)), 
                      palette = c("#2F4858", "#B01414"),
                      xlab = 'DFS Time (month)',
                      legend.title = '',
                      legend.labs = c('Low risk', 'High risk'))
  p_val <- ifelse(DFS_pval<0.001, '<0.001', sprintf("%.3f", DFS_pval))
  HR = paste0("HR = ", 
              sprintf("%.2f (%.2f-%.2f)", 
                      DFS_HR,
                      DFS_low95,
                      DFS_up95))
  print('----------------------------')
  print("AC0")
  print(p_val)
  print(HR)
  print('#############################')
  print(DFS_P)
  combined_plot <- plot_grid(DFS_P$plot, DFS_P$table,
                             nrow = 2, align = "v", axis = "lr",
                             rel_heights = c(5,1))
  save_path = paste0('F:/DKI/final_results/plot/new_KM/', 
                     cohort_name, '_', model_name, '_KM_AC0.tiff')
  ggsave(save_path, plot = combined_plot, device =  "tiff",
         width = 3900,height = 3000,units = 'px',dpi = 300)
}
######MT-NET comparison
library(scales)
library(ggradar)
MTnet_results <- read.csv('F:/DKI/final_results/excel_res/MTnet_results.csv')
MTnet_radar <- ggradar(MTnet_results,
        axis.labels = c("Dice", "AUC", "C-index"),
        axis.label.size = 5,
        axis.line.colour = "black",
        plot.title = "Comparison of performance with other multitasking models",
        legend.title = "Models",
        fill = TRUE,
        fill.alpha = 0.2) +
  theme(plot.title = element_text(size = 15),
        axis.text = element_blank(),
        axis.line.x = element_blank(),
        axis.line.y = element_blank(),
        legend.text = element_text(size = 13),
        legend.title = element_text(size = 14),
        legend.position = "top")+
  scale_color_manual(values = c("#0086FF", "#009D46", "#E64B35"),
                       name = 'Models',
                       breaks = c("TMSS", "DeepMTS", "Ours"),
                       labels = c("TMSS", "DeepMTS", "Ours"))+
  scale_fill_manual(values = c("#0086FF", "#009D46", "#E64B35"),
                    name = 'Models',
                    breaks = c("TMSS", "DeepMTS", "Ours"),
                    labels = c("TMSS", "DeepMTS", "Ours"))
MTnet_radar
ggsave('F:/DKI/final_results/plot/MTnet_radar.tiff', plot = MTnet_radar,
              width = 3900,height = 3000,units = 'px',dpi = 300)

ACC_table <- read.csv('F:/DKI/final_results/excel_res/ACC_table.csv')
int_acc <- ACC_table[c(1:6),-c(1)]
int_radar <- ggradar(int_acc,
        axis.label.size = 5,
        axis.line.colour = "black",
        legend.title = "Models",
        fill = TRUE,
        fill.alpha = 0.2) +
  theme(plot.title = element_text(size = 15),
        axis.text = element_blank(),
        axis.line.x = element_blank(),
        axis.line.y = element_blank(),
        legend.text = element_text(size = 13),
        legend.title = element_text(size = 14),
        legend.position = "right")+
  scale_color_manual(values = c("#E64B35", "#4DBBD5", "#E6C02B",
                                "#8700FF", "#147A5C", "#800000"),
                     name = 'Models',
                     breaks = c("All Model", "Clinical Model", "Preclinical Model",
                                "T2 Model", "T2+DKI Model", "T2+DKI+Preclinical Model"),
                     labels = c("All Model", "Clinical Model", "Preclinical Model",
                                "T2 Model", "T2+DKI Model", "T2+DKI+Preclinical Model"))+
  scale_fill_manual(values = c("#E64B35", "#4DBBD5", "#E6C02B",
                               "#8700FF", "#147A5C", "#800000"),
                    name = 'Models',
                    breaks = c("All Model", "Clinical Model", "Preclinical Model",
                               "T2 Model", "T2+DKI Model", "T2+DKI+Preclinical Model"),
                    labels = c("All Model", "Clinical Model", "Preclinical Model",
                               "T2 Model", "T2+DKI Model", "T2+DKI+Preclinical Model"))
ggsave('F:/DKI/final_results/plot/int_acc_RadarPlot.tiff', plot = int_radar,
       width = 3900,height = 3000,units = 'px',dpi = 300)

ext_acc <- ACC_table[c(7:12),-c(1)]
ext_radar <- ggradar(ext_acc,
                     axis.label.size = 5,
                     axis.line.colour = "black",
                     legend.title = "Models",
                     fill = TRUE,
                     fill.alpha = 0.2) +
  theme(plot.title = element_text(size = 15),
        axis.text = element_blank(),
        axis.line.x = element_blank(),
        axis.line.y = element_blank(),
        legend.text = element_text(size = 13),
        legend.title = element_text(size = 14),
        legend.position = "right")+
  scale_color_manual(values = c("#E64B35", "#4DBBD5", "#E6C02B",
                                "#8700FF", "#147A5C", "#800000"),
                     name = 'Models',
                     breaks = c("All Model", "Clinical Model", "Preclinical Model",
                                "T2 Model", "T2+DKI Model", "T2+DKI+Preclinical Model"),
                     labels = c("All Model", "Clinical Model", "Preclinical Model",
                                "T2 Model", "T2+DKI Model", "T2+DKI+Preclinical Model"))+
  scale_fill_manual(values = c("#E64B35", "#4DBBD5", "#E6C02B",
                               "#8700FF", "#147A5C", "#800000"),
                    name = 'Models',
                    breaks = c("All Model", "Clinical Model", "Preclinical Model",
                               "T2 Model", "T2+DKI Model", "T2+DKI+Preclinical Model"),
                    labels = c("All Model", "Clinical Model", "Preclinical Model",
                               "T2 Model", "T2+DKI Model", "T2+DKI+Preclinical Model"))
ggsave('F:/DKI/final_results/plot/ext_acc_RadarPlot.tiff', plot = ext_radar,
       width = 3900,height = 3000,units = 'px',dpi = 300)

Train_results$CMS4 <- factor(Train_results$CMS4, levels = c(0,1),
                             labels = c('CMS1-3', 'CMS4'))
ggplot(Train_results, aes(x=reorder(ID, merge_rad), y= merge_rad-0.5, fill=CMS4))+ 
  geom_bar(stat='identity')+
  scale_fill_manual( values = c("#2EC4B6","#202A36")) + 
  labs(y = 'Probability', x = NULL, title = 'Predicted probability of CMS4 in Training cohort')+
  theme(axis.ticks = element_blank(), axis.text.x = element_blank(),
        panel.grid = element_blank(), panel.background = element_blank(),
        title = element_text(face = "bold", size = 14, color = "black"), 
        axis.text = element_text(face = "bold", size = 11, color = "black"),
        axis.title.x = element_text(face = "bold", size = 14, color = "black", 
                                    margin = margin(c(15, 0, 0, 0))),
        axis.title.y = element_text(face = "bold", size = 14, color = "black", 
                                    margin = margin(c(0, 15, 0, 0))),
        panel.grid.minor = element_blank(),panel.grid.major = element_blank(),
        legend.title = element_blank(),legend.position = c(0.1,0.85),
        legend.text = element_text(face = "bold", size = 14, color = "black" ))+
  scale_y_continuous(breaks = c(-0.50,-0.25,-0.00,0.25,0.50),
                     labels = c(0.00,0.25,0.50,0.75,1.00))