set table "1.table"; set format "%.5f"
set samples 500.0; set parametric; plot [t=0:4] [] [] log10(10**t),20*log10(abs(1)) + (t<log10(1/0.5)?20*log10(1):+20*log10(1*0.5)+20*log10(10**t)) + (t<log10(1/(0.05))?20*log10(1):+20*log10(1/(0.05))-20*log10(10**t)) + (t<log10(1/0.002)?20*log10(1):+20*log10(1*0.002)+20*log10(10**t)) + (t<log10(1/(0.0002))?20*log10(1):+20*log10(1/(0.0002))-20*log10(10**t))
