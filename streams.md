|stream1|stream2|stream3|
|-----------------------------------|-----------------------------------------|------------------------------------|
|   measurementcopy                 | A\*result ->result                      | A\*process_error -> process_error  |
|                                   | Result + control -> result              | process_error \*At -> process_error|
|                                   | H*result->residual                      |                                    |
|                                   | Sync stream1                            |                                    |
|                                   | measurement - residual -> residual      | process_error + Q -> process_error |
|                                   | residual\*residualT->temp2              | process_error\*Ht -> kalman_gain   |
| Sync stream3                      |                                         |                                    |
| H\*kalman_gain -> temp            | copyTemp2 ->Innovationbank              |                                    |
| temp+R-> temp                     | Mean(innovation_bank)-> temp2           |                                    |
| invert temp                       | kalman_gain_final*temp2->temp2          |                                    |
|                                   | temp2*kalman_gain_finalT->process_noise |                                    |
| syncDevice                        |                                         |                                    |
| kalmanGain*temp->kalmanGain_Final |                                         |                                    |
|                                   | Sync Stream1                            |                                    |
| KalmanGainFinal*residual->temp    | Kalman_gain_Final\*H->temp2             |                                    |
| temp+result->result               | I-Temp2->temp2                          |                                    |
| Copy result                       | temp2*process_error->process_error      |                                    |


