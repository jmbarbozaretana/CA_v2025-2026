set table "2.table"; set format "%.5f"
set samples 500.0; set parametric; plot [t=0:1.7] [] [] log10(10**t),(t<log10(1/0.2)?20*log10(1):+20*log10(1*0.2)+20*log10(10**t))
