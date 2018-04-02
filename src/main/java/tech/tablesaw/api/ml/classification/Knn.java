/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package tech.tablesaw.api.ml.classification;

import com.google.common.base.Preconditions;
import smile.classification.KNN;
import tech.tablesaw.api.IntConvertibleColumn;
import tech.tablesaw.api.NumberColumn;
import tech.tablesaw.util.DoubleArrays;

import java.util.SortedSet;
import java.util.TreeSet;

public class Knn extends AbstractClassifier {

    private final KNN<double[]> classifierModel;

    private Knn(KNN<double[]> classifierModel) {
        this.classifierModel = classifierModel;
    }

    public static Knn learn(int k, IntConvertibleColumn labels, NumberColumn... predictors) {
        KNN<double[]> classifierModel = KNN.learn(DoubleArrays.to2dArray(predictors), labels.asIntArray(), k);
        return new Knn(classifierModel);
    }

    public int predict(double[] data) {
        return classifierModel.predict(data);
    }

    public ConfusionMatrix predictMatrix(IntConvertibleColumn labels, NumberColumn... predictors) {
        Preconditions.checkArgument(predictors.length > 0);

        SortedSet<Object> labelSet = new TreeSet<>(labels.asIntegerSet());
        ConfusionMatrix confusion = new StandardConfusionMatrix(labelSet);

        populateMatrix(labels.asIntArray(), confusion, predictors);
        return confusion;
    }

    public ConfusionMatrix predictMatrix(NumberColumn labels, NumberColumn... predictors) {
        Preconditions.checkArgument(predictors.length > 0);

        SortedSet<Object> labelSet = new TreeSet<>(labels.asIntegerSet());   // TODO NEEDS TO BE an INT SET
        ConfusionMatrix confusion = new StandardConfusionMatrix(labelSet);

        populateMatrix(labels.asIntArray(), confusion, predictors);
        return confusion;
    }

    public int[] predict(NumberColumn... predictors) {
        Preconditions.checkArgument(predictors.length > 0);
        int[] predictedLabels = new int[predictors[0].size()];
        for (int row = 0; row < predictors[0].size(); row++) {
            double[] data = new double[predictors.length];
            for (NumberColumn predictor : predictors) {
                data[row] = predictor.get(row);
            }
            predictedLabels[row] = classifierModel.predict(data);
        }
        return predictedLabels;
    }

    @Override
    int predictFromModel(double[] data) {
        //TODO(lwhite): Better tests
/*
        if (data[0] == 5.0)
            System.out.println(Arrays.toString(data));
*/
        return classifierModel.predict(data);
    }
}
