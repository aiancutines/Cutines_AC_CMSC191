$Console:Only
'============== backpropagation ==============
' uplb/cas/ics/cmsc191:NeuralComputing   backprop.bas by jppabico 2023.11.29
' Modified version.  Algorithm (forward, back, change, logistic) is unchanged.
' Changes:
'   1. Settings come from network.txt (one value per line):
'        hidden nodes, kappa, phi, theta, mu, direct(0/1), max epochs, patience, seed,
'        split (used by test.py only), restore best weights (1/0, optional)
'   2. PROBLEM holds train + validation + test rows (made by test.py):
'        ins / outs / nTrain / nValid / nTest, then per example:
'        the inputs line, the target line, the example-weight line
'   3. Example weights multiply the error derivative (class balancing);
'      derivatives are averaged over the training examples
'   4. Early stopping on the validation rows; best weights are restored
'      (network.txt line 11 = 0 keeps the FINAL weights instead)
'   5. Writes predictions.csv (validation + test) and trainlog.csv
' ============================================

Dim ins As Integer, hids As Integer, outs As Integer
Dim i As Integer, j As Integer, k As Integer
Dim m As Long, n As Long
Dim nTr As Long, nVa As Long, nTe As Long, total As Long
Dim direct As Integer
Dim seed As Long, maxEpochs As Long, patience As Long
Dim bad As Long, bestEpoch As Long
Dim restoreBest As Integer, splitDummy As Single
Dim trErr As Single, vaErr As Single, bestVal As Single
Dim sumWTr As Single, sumWVa As Single
Dim setName$

Dim Shared kappa As Single
Dim Shared phi As Single
Dim Shared theta As Single
Dim Shared mu As Single

Dim dataFile As String, prefix As String
On Error GoTo fatal
dataFile = "PROBLEM"
If Not _FileExists("network.txt") Then
    Print "network.txt NOT FOUND in "; _CWD$
    GoTo waitEnd
End If
Open "network.txt" For Input As #4
Input #4, hids
Input #4, kappa
Input #4, phi
Input #4, theta
Input #4, mu
Input #4, direct
Input #4, maxEpochs
Input #4, patience
Input #4, seed
restoreBest = 1
If Not EOF(4) Then Input #4, splitDummy 'line 10 (split) is used by test.py only
If Not EOF(4) Then Input #4, restoreBest 'line 11: 1 = restore best-validation weights, 0 = keep final weights
Close #4
If Not _FileExists(dataFile) Then
    Print "PROBLEM NOT FOUND in "; _CWD$
    Print "Run   python test.py   first, in this same folder."
    GoTo waitEnd
End If
Print "hidden="; hids; " kappa="; kappa; " phi="; phi; " theta="; theta; " mu="; mu; " direct="; direct

'--- Hyperparameters of the course program ---
Randomize seed

'--- Load data ---
Open dataFile For Input As #1
Input #1, ins
Input #1, outs
Input #1, nTr
Input #1, nVa
Input #1, nTe
total = nTr + nVa + nTe
Dim x(ins, total) As Single 'Inputs for examples
Dim t(outs, total) As Single 'Target outputs
Dim sw(total) As Single 'Example weights
For n = 1 To total
    For i = 1 To ins
        Input #1, x(i, n)
    Next i
    For k = 1 To outs
        Input #1, t(k, n)
    Next k
    Input #1, sw(n)
Next n
Close #1
sumWTr = 0!
For n = 1 To nTr
    sumWTr = sumWTr + sw(n)
Next n
sumWVa = 0!
For n = nTr + 1 To nTr + nVa
    sumWVa = sumWVa + sw(n)
Next n

'--- Memory allocation ---
Dim a(0 To ins, 1 To hids) As Single 'Hidden weights
Dim b(0 To hids + ins, 1 To outs) As Single 'Output weights
Dim bestA(0 To ins, 1 To hids) As Single
Dim bestB(0 To hids + ins, 1 To outs) As Single
Dim cHid(0 To ins, 1 To hids) As Single
Dim cOut(0 To hids + ins, 1 To outs) As Single
Dim dHid(0 To ins, 1 To hids) As Single
Dim dOut(0 To hids + ins, 1 To outs) As Single
Dim eHid(0 To ins, 1 To hids) As Single
Dim eOut(0 To hids + ins, 1 To outs) As Single
Dim fHid(0 To ins, 1 To hids) As Single
Dim fOut(0 To hids + ins, 1 To outs) As Single
Dim u As Single
Dim y(hids) As Single
Dim v As Single
Dim z(outs) As Single
Dim p(outs) As Single
Dim qv(hids) As Single

GoSub initWeights

Open "trainlog.csv" For Output As #3
Print #3, "epoch,train_wmse,val_wmse"

bestVal = 1E+30
bestEpoch = 0
bad = 0
m = 0
'=== Main Program ===
Do
    m = m + 1
    trErr = 0!
    For n = 1 To nTr
        GoSub forward
        GoSub back
        For k = 1 To outs
            trErr = trErr + sw(n) * (z(k) - t(k, n)) ^ 2
        Next k
    Next n
    GoSub changeWeights
    trErr = trErr / sumWTr

    vaErr = 0!
    For n = nTr + 1 To nTr + nVa
        GoSub forward
        For k = 1 To outs
            vaErr = vaErr + sw(n) * (z(k) - t(k, n)) ^ 2
        Next k
    Next n
    vaErr = vaErr / sumWVa

    Write #3, m, trErr, vaErr

    If vaErr < bestVal - 0.000001! Then
        bestVal = vaErr
        bestEpoch = m
        bad = 0
        For j = 1 To hids
            For i = 0 To ins
                bestA(i, j) = a(i, j)
            Next i
        Next j
        For k = 1 To outs
            For j = 0 To hids + ins
                bestB(j, k) = b(j, k)
            Next j
        Next k
    Else
        bad = bad + 1
    End If

    If m Mod 100 = 0 Then
        Print "Epoch:"; m; " train:"; trErr; " val:"; vaErr; " best epoch:"; bestEpoch
    End If

    If bad >= patience Or m >= maxEpochs Then Exit Do
Loop
Close #3

'--- Restore the best weights (unless network.txt line 11 is 0) ---
If restoreBest <> 0 Then
    For j = 1 To hids
        For i = 0 To ins
            a(i, j) = bestA(i, j)
        Next i
    Next j
    For k = 1 To outs
        For j = 0 To hids + ins
            b(j, k) = bestB(j, k)
        Next j
    Next k
End If

'--- Write validation + test predictions ---
Open "predictions.csv" For Output As #2
Print #2, "set,target,score"
For n = nTr + 1 To total
    GoSub forward
    If n <= nTr + nVa Then setName$ = "V" Else setName$ = "T"
    Write #2, setName$, t(1, n), z(1)
Next n
Close #2
Print "DONE epochs="; m; " best_epoch="; bestEpoch; " best_val="; bestVal
Print "Now run:  python test.py score"
GoTo waitEnd

fatal:
Print "RUNTIME ERROR"; Err; "at line"; _ErrorLine

waitEnd:
Print "Press Enter to close"
Dim dummy As String
Input dummy
End

'--- Subroutines ---

initWeights:
For j = 1 To hids
    For i = 0 To ins
        a(i, j) = 0.2 * (Rnd - 0.5)
        eHid(i, j) = kappa
    Next i
Next j
For k = 1 To outs
    For j = 0 To hids
        If j Mod 2 = 0 Then
            b(j, k) = 1!
        Else
            b(j, k) = -1!
        End If
        eOut(j, k) = kappa
    Next j
    If direct Then
        For j = hids + 1 To hids + ins
            b(j, k) = 0.2 * (Rnd - 0.5)
            eOut(j, k) = kappa
        Next j
    End If
Next k
Return

forward:
For j = 1 To hids
    u = a(0, j)
    For i = 1 To ins
        u = u + (a(i, j) * x(i, n))
    Next i
    y(j) = logistic(u)
Next j
For k = 1 To outs
    v = b(0, k)
    For j = 1 To hids
        v = v + (b(j, k) * y(j))
    Next j
    If direct Then
        For j = hids + 1 To hids + ins
            v = v + (b(j, k) * x(j - hids, n))
        Next j
    End If
    z(k) = logistic(v)
Next k
Return

back:
For j = 1 To hids
    qv(j) = 0!
Next j
For k = 1 To outs
    'weighted, averaged derivative of the squared error w.r.t. v
    p(k) = (z(k) - t(k, n)) * z(k) * (1! - z(k)) * sw(n) / nTr
    dOut(0, k) = dOut(0, k) + p(k)
    For j = 1 To hids
        dOut(j, k) = dOut(j, k) + p(k) * y(j)
        qv(j) = qv(j) + p(k) * b(j, k)
    Next j
    If direct Then
        For j = hids + 1 To hids + ins
            dOut(j, k) = dOut(j, k) + (p(k) * x(j - hids, n))
        Next j
    End If
Next k
For j = 1 To hids
    qv(j) = qv(j) * y(j) * (1! - y(j))
    dHid(0, j) = dHid(0, j) + qv(j)
    For i = 1 To ins
        dHid(i, j) = dHid(i, j) + qv(j) * x(i, n)
    Next i
Next j
Return

changeWeights:
For j = 1 To hids
    For i = 0 To ins
        Call change(a(i, j), dHid(i, j), fHid(i, j), eHid(i, j), cHid(i, j))
    Next i
Next j
For k = 1 To outs
    For j = 0 To hids
        Call change(b(j, k), dOut(j, k), fOut(j, k), eOut(j, k), cOut(j, k))
    Next j
    If direct Then
        For j = hids + 1 To hids + ins
            Call change(b(j, k), dOut(j, k), fOut(j, k), eOut(j, k), cOut(j, k))
        Next j
    End If
Next k
For j = 1 To hids
    For i = 0 To ins
        dHid(i, j) = 0!
    Next i
Next j
For k = 1 To outs
    For j = 0 To hids + ins
        dOut(j, k) = 0!
    Next j
Next k
Return


Sub change (w, d, f, e, c)
    If d * f > 0! Then
        e = e + kappa
    Else
        e = e * phi
    End If
    f = (1! - theta) * d + theta * f
    c = (1! - mu) * -1! * e * d + mu * c
    w = w + c
End Sub


Function logistic (uv)
    If uv < -88! Then
        logistic = 0!
    ElseIf uv > 88! Then
        logistic = 1!
    Else
        logistic = 1! / (1! + Exp(-uv))
    End If
End Function
