-- William Hudgins
-- CSCI 437
-- quicksort.hs
-- 11/03/2014
--
-- Gaussian Elimination in Haskell


import Data.List
import Data.Maybe

main =
	do
		let matrixA = [[1071, 55, 34, 87, 74, 6, 37, 37, 75, 72, 68, -6, -9, 83, 36, 70, -1, 35, 0, 38, -17, 12, -18, 37, 66, 27, 82, 4, 14, 17, 0],
			[84, 1261, 23, 18, 55, 14, -10, 45, 41, 76, 19, 62, 88, 65, -3, 84, 5, 49, -14, 60, 88, 31, 24, -12, 22, 88, 28, 23, 49, -7, 89],
			[1, 66, 1286, 36, 46, 40, 23, 20, 72, 15, 56, 47, 63, 23, 15, -19, 75, 59, 67, 65, 77, 13, 88, 57, 49, 87, -8, -1, 75, 47, 8],
			[42, 53, 86, 1179, 22, 29, 40, 6, 12, 90, -7, -9, 2, 37, 26, 75, 41, 39, 5, 46, 71, 59, 81, 87, 52, 11, -17, 59, 21, 60, -18],
			[45, 39, 59, 24, 1523, 65, 70, 37, 63, 10, -9, 57, 9, 32, 66, 30, 59, 20, 74, -4, 65, 61, 34, 42, 55, 79, 70, 62, 89, 80, 31],
			[65, 18, 41, -2, 11, 1219, 84, 21, -16, 35, 69, 68, 1, 89, -10, 58, 7, 16, 47, 28, 39, 39, 66, 77, 61, 76, -10, -1, 13, 80, 73],
			[35, 42, 56, -15, 61, 37, 1182, 8, 78, 71, 59, 3, 17, 27, 36, 63, 2, 70, 68, 89, 88, 47, 0, 2, -16, 19, 63, 17, 0, 64, -11],
			[82, -7, 74, 41, 0, 45, 85, 1054, 63, 83, -12, 80, 5, 56, 79, 33, -5, 18, 78, 10, 41, -2, 6, 4, -17, 2, -4, 86, 24, 35, 55],
			[76, 48, 60, 64, 82, 14, 37, 13, 1192, 45, 31, 42, 36, 4, 77, 0, 84, -16, 76, 8, 57, 68, 85, 85, 24, -16, 13, 53, 19, -13, -2],
			[-5, 59, 29, 33, 79, -6, -1, 37, 50, 995, 65, 6, 40, 13, 50, 20, 70, 24, -15, 1, 35, 89, -1, 24, 90, -1, 86, 17, 28, -13, 43],
			[28, 84, 29, 82, 71, 58, 42, 4, 69, 82, 1560, -15, 31, 56, 66, 87, -17, 72, 84, 44, 16, 75, 54, 41, 74, 69, 26, 80, -16, 27, 72],
			[79, -12, 64, 39, 35, -14, -7, 28, 88, 78, -9, 1097, 84, 38, 85, 0, -2, 83, 41, 72, -4, 46, -5, 72, 9, 19, 1, 73, 55, -11, 22],
			[88, 43, -19, 50, 50, 84, 66, 37, -13, 79, 60, 71, 1186, 46, 78, 5, 64, 0, 12, -14, 35, 76, -20, 47, 43, 5, 11, -15, 11, 42, 61],
			[85, 51, 89, 90, 28, -11, -14, 41, 66, 66, 30, 80, 25, 1397, 73, 23, 34, 38, -11, 18, 84, 83, 73, 5, 23, 70, 22, 19, 21, 27, 87],
			[53, 4, 17, -9, 90, 54, 7, 3, 39, 89, 56, 60, 17, -4, 1178, 77, 28, 34, 39, 70, 79, 87, 2, 21, 7, 71, 35, 1, 35, 73, 31],
			[-20, 33, 47, 60, 27, -20, -14, -18, 39, 30, 18, -16, -6, -14, 51, 841, 33, 89, 26, -10, 27, 39, 41, 29, 77, 85, 53, -13, 19, 67, 45],
			[58, -9, 55, 33, 14, 26, -1, 64, 87, 82, 1, 40, -16, 23, 54, 4, 1009, 8, 52, 76, 34, -4, -13, 69, 59, 58, 78, 6, 0, 23, 21],
			[15, 40, 63, 22, 56, -13, 4, 39, 43, -2, 6, 21, 57, 1, 89, 29, 67, 1045, 7, 51, 6, 47, 36, -5, -17, 85, 50, 21, 75, 50, 10],
			[40, 5, -17, 49, 83, 23, 8, 51, 34, 51, 85, -15, 19, -5, 12, -2, -18, 51, 688, -19, -6, 83, 5, 4, 4, -19, 83, 13, 0, -16, 46],
			[0, 20, 41, 5, 47, 77, -10, 6, 3, -20, -9, 90, 27, 69, 0, 3, 55, 9, 35, 1058, 58, 64, 69, 0, 32, 35, 45, 47, 77, 59, 28],
			[-18, 40, -19, 47, -16, 49, 25, 42, 45, 51, 52, 5, -17, 24, 37, 66, 82, 49, 19, 36, 1047, 10, 84, 50, 68, 4, 71, 6, 62, 18, 9],
			[83, 0, 56, -3, 0, 35, 49, -7, 69, 69, 50, 26, -9, 27, 61, -10, 39, -14, 76, 16, 1, 1037, 19, 12, 60, 27, 20, 76, 27, 11, 73],
			[3, 31, 52, 76, 89, 81, -1, 71, 69, 39, 2, 52, 19, -9, -6, 31, 72, 51, 78, 54, 58, 88, 1380, 32, 25, 37, 2, 1, 49, 77, 56],
			[30, 60, 16, 41, 60, -9, 24, 33, 53, 50, -8, 21, 75, 8, 38, 20, 36, -4, 31, 47, 79, 13, 77, 1054, 55, 9, 1, 18, 89, 86, -2],
			[81, 48, 13, 18, 88, 39, -16, 35, -9, 24, 89, -18, 18, 0, -10, 70, 33, 43, -12, 41, 75, 13, -8, 30, 987, -3, 19, 72, 46, 50, 48],
			[-18, 59, 7, 68, 19, 79, 20, 0, -18, 49, 26, 74, 67, 86, 8, 9, 78, 83, 40, 10, 34, 38, 84, 5, 87, 1188, 22, -14, 55, 10, 22],
			[11, 42, 6, 39, 51, 45, 0, 82, 50, 12, 38, 45, 59, -5, 66, -6, 19, 42, 80, 77, 70, 21, 46, 52, 41, -6, 1225, 52, 72, 16, 49],
			[19, 4, 29, 32, 89, 81, 2, 78, 57, 38, 58, 78, -16, -18, 63, 5, 74, 48, 90, 36, -7, 31, 56, -12, 19, 33, 74, 1152, -8, 30, 15],
			[62, 90, -18, 22, -2, 79, 77, 11, -19, 54, 23, 61, 34, -1, 29, 22, 32, 52, 8, 58, 40, 20, -11, 87, 32, 55, 73, 48, 1198, -16, 90],
			[11, 34, -3, 67, 60, 3, 70, 86, -9, 75, 74, 85, -3, 48, 74, 88, 2, 78, 73, -12, 83, 64, 60, 44, 70, 43, 45, 78, 60, 1585, 45]]
		let solutionVector = gaussianElim matrixA
		print "Starting matrix:"
		mapM_ print matrixA
		print "Solution vector after Gaussian elimination:"
		mapM_ print solutionVector

-- Find largest value in the first column
findPivot :: [[Double]] -> Int
findPivot matrixA =
	let
		firstCol = (transpose(matrixA)!!0)
		maxColValue = maximum firstCol
		pivotIndex = elemIndex maxColValue firstCol
	in fromJust(pivotIndex)

addRows :: [Double] -> [Double] -> [Double]
addRows pivotRow targetRow = 
	let 
		targetValue = targetRow!!0
		pivotValue = pivotRow!!0
		multiplier = (targetValue/pivotValue)
		multipliedPivotRow = map (* multiplier) pivotRow
		modifiedTargetRow = zipWith (-) multipliedPivotRow targetRow
	in map (*(0-1)) modifiedTargetRow

switchRows :: [[Double]] -> Int -> [[Double]]
switchRows matrixA pivotIndex = (drop pivotIndex matrixA) ++ (take pivotIndex matrixA)

forwardElimination :: [[Double]] -> [[Double]]
forwardElimination matrixA =
	let
		pivotIndex = findPivot matrixA
		pivotedMatrixA = switchRows matrixA pivotIndex
		pivotRow = (pivotedMatrixA !! 0)
		subMatrix = drop 1 pivotedMatrixA -- removes pivot row
		subMatrix2 = map (addRows pivotRow) subMatrix -- add pivot row to others
		fixedCol = map (take 1) subMatrix2 -- Take first column, it is fixed
		newSubMatrix = map (drop 1) subMatrix2 --- removes columns
		compositeMatrix = if (length newSubMatrix > 0) then let
			newSubMatrix2 = forwardElimination newSubMatrix
			compositeSubMatrix = zipWith (++) fixedCol newSubMatrix2  -- Readd fixed col
			in ([matrixA !! 0]  ++ compositeSubMatrix) -- put first row into composite matrix 
		else 
			[pivotRow] ++ subMatrix
		in compositeMatrix

backSubstitution :: [[Double]] -> [Double] -> [Double] -> [Double]
backSubstitution matrixA vectorB vectorX =
	let
		variablesToPlugIn = length vectorX
		coeffs = head matrixA
		rh = head vectorB
		rowSum = if (variablesToPlugIn > 0) then
			sum $ zipWith (*) (take variablesToPlugIn coeffs) vectorX
		else
			0
		rhModified = rh - rowSum
		solution = rhModified / (coeffs!!variablesToPlugIn)
		finalVectorX = if (length matrixA > 1) then
			backSubstitution (drop 1 matrixA) (drop 1 vectorB) (vectorX ++ [solution])
		else 
			vectorX ++ [solution]
	in finalVectorX

gaussianElim :: [[Double]] -> [Double]
gaussianElim matrixA = 
	let 
		rowEchelonMatrixA = forwardElimination matrixA 
		numCoeffs = (length (rowEchelonMatrixA!!0)) - 1
		leftHandMatrixA = map (take numCoeffs) rowEchelonMatrixA
		leftHandMatrixA2 = reverse $ map (reverse) leftHandMatrixA
		rightHandMatrixA = reverse $ concat (map (drop numCoeffs) rowEchelonMatrixA)
	in reverse $ backSubstitution leftHandMatrixA2 rightHandMatrixA []

