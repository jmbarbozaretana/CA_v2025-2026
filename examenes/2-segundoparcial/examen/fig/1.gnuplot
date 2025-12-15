set table "1.table"; set format "%.5f"
set samples 500.0; set parametric; plot [t=-1:4] [] [] log10(10**t),-20*log10(abs(1/(10**t))) + (t<log10(1/(0.125))?20*log10(40):+20*log10(40/(0.125))-20*log10(10**t)) + (t<log10(1/(0.003125))?20*log10(1):+20*log10(1/(0.003125))-20*log10(10**t))
