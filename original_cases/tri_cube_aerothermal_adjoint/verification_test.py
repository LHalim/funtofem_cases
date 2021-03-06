#test.py

import numpy as np

# Initialize
cq_mag_orig = [-3.512681257225101e-05, -3.781997266723326e-05, -1.878399673523809e-05, -1.987024319593699e-05, -1.786233617777174e-05, -3.285631005749469e-05, -3.212455071500637e-05, -2.614198397243643e-05, -1.561913940109100e-05, -1.112445427988957e-05, -1.330052864391299e-05, -2.182215805322324e-05, 8.519245424568513e-06,  3.402392733661417e-05, 4.331640233235250e-05, -2.161381834351192e-05, -8.557433997796686e-05, -2.600759476083527e-05, -4.139573620871771e-05, -1.538852963249822e-05, -5.513756179833820e-06, -2.352163759835009e-05, -5.880652744429190e-05, -2.345520476239892e-05, -3.123978150657387e-05, -1.070700682720567e-05, -1.246854507921375e-05, -3.081333293267168e-05, -3.954409546071492e-05, -4.171992923345020e-05, -4.176692683204349e-05, -1.463809346087124e-05, -2.952911310214296e-05]
cq_mag_perturb = [-3.512729474431120e-05, -3.782061202233749e-05, -1.878481990898604e-05, -1.987123595877095e-05, -1.786299044418832e-05, -3.285719736957944e-05, -3.212545163341264e-05, -2.614292947801188e-05, -1.561982578684398e-05, -1.112520629061797e-05, -1.330115883705244e-05, -2.182298113514973e-05, 8.518467108841565e-06, 3.402256646854336e-05,  4.331518279818880e-05, -2.161454399668882e-05, -8.557551774576685e-05, -2.600941778512157e-05, -4.139755668303900e-05, -1.539030603167194e-05, -5.516026603316808e-06, -2.352284026465322e-05, -5.880797373518751e-05, -2.345655613587481e-05, -3.124161822434638e-05, -1.070916725401419e-05, -1.246962647403209e-05, -3.081528890667499e-05, -3.954543797118287e-05, -4.172096616192751e-05, -4.176908789031845e-05, -1.463938545762066e-05, -2.953107174476889e-05]

cq_mag_orig = np.array(cq_mag_orig)
cq_mag_perturb = np.array(cq_mag_perturb)

E = 1e-6

lam_mag_thermal = [[3.468831010565972e+00, 5.518172944703251e-01, 4.901744019511799e-05], [3.513284260095697e+00, 5.590263919354738e-01, 4.967012306093293e-05], [3.530876009703484e+00, 5.617200858777647e-01, 4.992198639810477e-05], [3.513280276032266e+00, 5.585761736913755e-01, 4.965159225845645e-05], [3.468824146159245e+00, 5.510413486541842e-01, 4.898550297237471e-05], [3.492477744006948e+00, 5.546674037870742e-01, 4.931711130335088e-05], [3.483786064522599e+00, 5.527399432539665e-01, 4.915476719152148e-05], [3.392091108811085e+00, 5.368934001007725e-01, 4.774751214029282e-05], [3.183671366299491e+00, 5.019789898823195e-01, 4.462881491815950e-05], [3.211107246891511e+00, 5.057134648618221e-01, 4.495320400808008e-05], [3.230738493373771e+00, 5.090403671705761e-01, 4.521018615083308e-05], [3.211122702084507e+00, 5.074663775178392e-01, 4.502534884125706e-05], [3.183690186441858e+00, 5.041125844465917e-01, 4.471662556345337e-05], [3.392108967048128e+00, 5.389143278313745e-01, 4.783069186012144e-05], [3.483799859839052e+00, 5.542996299356687e-01, 4.921896365473603e-05], [3.492487510125537e+00, 5.557712853185363e-01, 4.936254659727178e-05], [3.535904509752812e+00, 5.628247289145772e-01, 5.000234361098572e-05], [3.435955168198916e+00, 5.435705490193584e-01, 4.834055891451318e-05], [3.543003843803215e+00, 5.632322674734963e-01, 5.007502879666642e-05], [3.457369168812090e+00, 5.490154658514563e-01, 4.873894903228657e-05],  [3.574451634640199e+00, 5.689197988199544e-01, 5.053294578082268e-05], [3.549315615306588e+00, 5.650461589525931e-01, 5.019219203626232e-05], [3.596012686052293e+00, 5.725522426561277e-01, 5.087620905116143e-05], [3.504233961521558e+00, 5.554340735483807e-01, 4.934433278284064e-05], [3.551898711336803e+00, 5.637514699276868e-01, 5.013754322713545e-05], [3.616427735902618e+00, 5.743380059541058e-01, 5.106590212035058e-05], [3.448468340721472e+00, 5.452934053625260e-01, 4.847378776478987e-05], [3.608660816373576e+00, 5.738290261309329e-01, 5.102255951646166e-05], [3.547484354079676e+00, 5.635828795709382e-01, 5.011559273198822e-05], [3.561445831371937e+00, 5.668691647836496e-01, 5.037444810624629e-05], [3.588206334461436e+00, 5.709841053158988e-01, 5.075474132268787e-05], [3.608762955800114e+00, 5.738014810334756e-01, 5.098258290439531e-05], [3.641529714072004e+00, 5.795949283088011e-01, 5.151545085022540e-05]]
lam_t_temp = [-1.707739747748015e-07,  2.713080760303352e-07,  2.336519078281304e-07,  1.918043638547144e-07,  4.097244546203057e-07,  4.775549111195875e-07,  5.399119337984008e-07,  4.910287964823809e-07,  3.953918260037725e-07,  1.284009837431295e-07,  1.011981295714578e-07,  2.465554827523281e-07,  3.363337549389251e-07,  8.847072814811655e-07,  1.748757242411774e-06,  2.724978207150973e-08,  4.594750061099118e-07,  3.519641923493640e-07,  3.696373557881543e-07,  2.980322697109999e-07,  1.859423279037406e-07,  2.188144691047741e-07,  4.764603865534264e-07, 3.559787075958356e-07,  3.673175552472723e-07,  5.190419765234103e-07,  3.154817196835186e-07,  4.444298909436158e-07,  3.442510515015194e-07, 2.863647617857842e-07,  6.597099476535496e-07,  4.290047405895799e-07,  5.071893764968246e-07]

# Verification Test
output1 = np.transpose(cq_mag_perturb - cq_mag_orig) / E
output1 = np.matmul(output1, lam_mag_thermal )
output2 = lam_t_temp
#error = abs(output1-output2)

print('output1 = ', output1)
print('output2 = ', output2)
#print('Error = ', error)
