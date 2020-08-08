using LinearAlgebra
msize =64
numofmeasurements = 100
measurements = ones(msize,msize,numofmeasurements) - rand(-.1:.0001:.1,msize,msize,numofmeasurements)
iden = diagm(repeat([1],msize))
consts = vec(hcat(iden,zeros(msize,msize),iden,iden./20,iden,iden))
#consts = rand(msize*msize*6)
zerosinit = zeros(msize*msize)

file = open("data/largeMeasurements.hpp","w+")
print(size(measurements))
write(file, "float measurements["*string(numofmeasurements)*"]["*string(msize*msize)*"] = {")
write(file, replace(replace(replace(replace(string(permutedims(measurements,(2,1,3))) , "[" => "{"),"]"=>"},")," "=>", "),";,"=>",")[1:end-1])  
write(file, "};\n")

write(file,"float batched_const_matrices["*string(msize*msize*6)*"] = {")
write(file,replace(replace(string(consts),"["=>""),"]"=>""))
write(file,"};\n")

write(file,"float mean_init["*string(msize*msize)*"] = {")
write(file,replace(replace(string(zerosinit),"["=>""),"]"=>""))
write(file,"};\n")
close(file)