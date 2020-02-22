from Digit import Digit
import numpy as np
from data_loader import*
def main():
    zeroTraining, zeroTrainingLengths, zeroTest, zeroTestLength, zeroTestLines = load_data(r"F:\MLDM\2nd Semester\Done_ML\Practical 2\digit_strings\0.txt")
    oneTraining, oneTrainingLengths, oneTest, oneTestLength, oneTestLines = load_data(r"F:\MLDM\2nd Semester\Done_ML\Practical 2\digit_strings\1.txt")
    twoTraining, twoTrainingLengths, twoTest, twoTestLength, twoTestLines = load_data(r"F:\MLDM\2nd Semester\Done_ML\Practical 2\digit_strings\2.txt")
    threeTraining, threeTrainingLengths, threeTest, threeTestLength, threeTestLines = load_data(r"F:\MLDM\2nd Semester\Done_ML\Practical 2\digit_strings\3.txt")
    fourTraining, fourTrainingLengths, fourTest, fourTestLength, fourTestLines = load_data(r"F:\MLDM\2nd Semester\Done_ML\Practical 2\digit_strings\4.txt")
    fiveTraining, fiveTrainingLengths, fiveTest, fiveTestLength, fiveTestLines = load_data(r"F:\MLDM\2nd Semester\Done_ML\Practical 2\digit_strings\5.txt")
    sixTraining, sixTrainingLengths, sixTest, sixTestLength, sixTestLines = load_data(r"F:\MLDM\2nd Semester\Done_ML\Practical 2\digit_strings\6.txt")
    sevenTraining, sevenTrainingLengths, sevenTest, sevenTestLength, sevenTestLines = load_data(r"F:\MLDM\2nd Semester\Done_ML\Practical 2\digit_strings\7.txt")
    eightTraining, eightTrainingLengths, eightTest, eightTestLength, eightTestLines = load_data(r"F:\MLDM\2nd Semester\Done_ML\Practical 2\digit_strings\8.txt")
    nineTraining, nineTrainingLengths, nineTest, nineTestLength, nineTestLines = load_data(r"F:\MLDM\2nd Semester\Done_ML\Practical 2\digit_strings\9.txt")
    numberZeroModel = Digit(name="Zero", n_states = 12)
    numberOneModel = Digit(name="One", n_states = 18)
    numberTwoModel = Digit(name="Two", n_states = 12)
    numberThreeModel = Digit(name="Three", n_states = 12)
    numberFourModel = Digit(name="Four", n_states = 13)
    numberFiveModel = Digit(name="Five", n_states = 12)
    numberSixModel = Digit(name="Six", n_states = 10)
    numberSevenModel = Digit(name="Seven", n_states = 8)
    numberEightModel = Digit(name="Eight", n_states = 15)
    numberNineModel = Digit(name="Nine", n_states = 13)

    models = [numberZeroModel, numberOneModel, numberTwoModel, numberThreeModel, numberFourModel, numberFiveModel, numberSixModel, numberSevenModel, numberEightModel, numberNineModel ]
    testSets = [zeroTestLines, oneTestLines, twoTestLines, threeTestLines, fourTestLines, fiveTestLines, sixTestLines, sevenTestLines, eightTestLines, nineTestLines]
    num_alltestCase = 0
    num_correctly_classified = 0
    for digitTestSet in range(0, len(testSets)):
        for testCase in testSets[digitTestSet]:
            num_alltestCase = num_alltestCase + 1
            maxProb = -np.inf
            maxModel = -1
            for i in range(0, len(models)):
                prob = models[i].prob(testCase, [len(testCase)])
                if prob > maxProb:
                    maxProb = prob
                    maxModel = i
            if maxModel == digitTestSet:
                num_correctly_classified = num_correctly_classified + 1


        print("Correct: ", num_correctly_classified, " out of: ", num_alltestCase)
        ratio = (num_correctly_classified / num_alltestCase)
        print("Total accuracy: ", ratio * 100, "% ")
        # Total accuracy:  87.35436203466894 % 

        # print("sequence class for dataset: ", testSetIndex, " is: ", models[maxModel].name)

        # print(models[i].name," : ", prob)

    # numberZeroModel.train(zeroTraining, zeroTrainingLengths)
    # numberZeroModel.save()
    
    # numberOneModel.train(oneTraining, oneTrainingLengths)
    # numberOneModel.save()
    
    # numberTwoModel.train(twoTraining, twoTrainingLengths)
    # numberTwoModel.save()
    
    # numberThreeModel.train(threeTraining, threeTrainingLengths)
    # numberThreeModel.save()
    
    # numberFourModel.train(fourTraining, fourTrainingLengths)
    # numberFourModel.save()
    
    # numberFiveModel.train(fiveTraining, fiveTrainingLengths)
    # numberFiveModel.save()
    
    # numberSixModel.train(sixTraining, sixTrainingLengths)
    # numberSixModel.save()
    
    # numberSevenModel.train(sevenTraining, sevenTrainingLengths)
    # numberSevenModel.save()
    
    # numberEightModel.train(eightTraining, eightTrainingLengths)
    # numberEightModel.save()

    # numberNineModel.train(nineTraining, nineTrainingLengths)
    # numberNineModel.save()
main()