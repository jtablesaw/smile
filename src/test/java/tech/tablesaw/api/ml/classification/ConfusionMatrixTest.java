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

import org.junit.Test;
import smile.classification.KNN;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.Table;
import tech.tablesaw.util.DoubleArrays;

import java.util.SortedSet;
import java.util.TreeSet;

public class ConfusionMatrixTest {

    @Test
    public void testAsTable() throws Exception {

        Table example = Table.read().csv("data/KNN_Example_1.csv");

        Table[] splits = example.sampleSplit(.5);
        Table train = splits[0];
        Table test = splits[1];

        KNN<double[]> knn = KNN.learn(
                DoubleArrays.to2dArray(train.nCol("X"), train.nCol("Y")),
                train.numberColumn(2).asIntArray(), 2);

        int[] predicted = new int[test.rowCount()];
        SortedSet<Object> lableSet = new TreeSet<>(train.numberColumn(2).asIntegerSet());
        ConfusionMatrix confusion = new StandardConfusionMatrix(lableSet);
        for (Row row : test) {
            double[] data = new double[2];
            data[0] = test.numberColumn(0).get(row.getRowNumber());
            data[1] = test.numberColumn(1).get(row.getRowNumber());
            predicted[row.getRowNumber()] = knn.predict(data);
            confusion.increment((int) test.numberColumn(2).get(row.getRowNumber()), predicted[row.getRowNumber()]);
        }
    }
}