fname <- 'quantiles.dat'
write('# GENERATED FILE, DO NOT EDIT.', fname)
write('# Quartiles calculated by R.', fname, append=TRUE)
write('# ==========================', fname, append=TRUE)
data <- seq(1000, 1999, 100)
write('seq = 1000 1999 100', fname, append=TRUE)
ps <- c(0.0, 0.01, 0.05, 0.1, 0.16, 0.2, 0.25, 0.31, 0.42, 0.45, 0.5, 0.55, 0.62, 0.75, 0.81, 0.87, 0.9, 0.95, 0.99, 1.0)
write(c('p = ', ps), fname, ncolumns= 22, append=TRUE)
for (N in c(1:9)) {
    write(c(N, ':', quantile(data, ps, type=N)), ncolumns=22, fname, append=TRUE)
    }


