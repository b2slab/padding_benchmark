#!/usr/bin/env Rscript

# this script cleans the non-optimal weights of the form
# weights-improvement-xxxx.hdf5 by keeping only the maximum
# accuracy in xxxx

dir.clean <- "~angela/padding_EBI/data/checkpoint/EC_number"
# dir.clean <- "task"
dirs <- list.dirs(dir.clean, recursive = TRUE)

best_weights <- function(vec.hdf5) {
  acc <- gsub(
    "(.*weights-improvement-\\d+-)([0-9\\.]+)(\\.hdf5$)",
    "\\2",
    vec.hdf5)
  accnum <- as.numeric(acc)
  
  vec.hdf5[which.max(accnum)]
}

while(TRUE) {
  message("Cleaning sub-optimal weights from ", dir.clean)
  
  tmp <- lapply(dirs, function(d) {
    print(d)
    hdf5 <- list.files(
      d,
      pattern = "weights-improvement-\\d+-[0-9\\.]+\\.hdf5$",
      full.names = TRUE,
      include.dirs = FALSE)
    
    if (length(hdf5) > 0) {
      # message("File names:")
      # print(hdf5)
      
      message("Best:")
      print(best_weights(hdf5))
      
      file.del <- setdiff(hdf5, best_weights(hdf5))
      message("Delete:")
      print(file.del)
      
      unlink(file.del)
      
      message("")
      
      invisible()
    }
  })
  
  message("Waiting to clean again...")
  Sys.sleep(300)
}

message("Done!")

