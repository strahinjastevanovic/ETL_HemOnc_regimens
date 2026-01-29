inspect_rda <- function(file_path) {
  # Create a temporary environment
  temp_env <- new.env()
  
  # Load the .rda file into the temporary environment
  object_names <- load(file_path, envir = temp_env)
  
  cat(sprintf("Contents of '%s':\n", file_path))
  
  # Iterate through the names of the loaded objects
  for (name in object_names) {
    obj <- get(name, envir = temp_env)
    
    # Check if the object has dimensions (like a data frame, matrix, or array)
    if (!is.null(dim(obj))) {
      cat(sprintf("* Object Name: '%s', Type: %s, Dimensions: [%d rows, %d columns]\n", 
                  name, class(obj), nrow(obj), ncol(obj)))
    } else {
      # For other objects like vectors, lists, etc.
      cat(sprintf("* Object Name: '%s', Type: %s, Length: %d\n", 
                  name, class(obj), length(obj)))
    }
  }
}

inspect_rda("/home/stev/proj/ETL_HemOnc_regimens/output.8/regimens.rda")
inspect_rda("/home/stev/proj/regimens/data/regimens.rda")