# ファイル構造
```
├── data：実験で利用するデータ
    ├── Hazumi：GitHubからダウンロードしたHazumiデータ
    ├── Hazumi_features：src/feature_extraction.ipynbを実行して実験で利用する形式に変換されたHazumiデータ
├── robocom：ロボットコンペ2023に利用した心象推定器の学習コードとモデルデータ
├── src：ユーザ特性を考慮した心象推定に関する実験コード
```

# セットアップ
- Anacondaの仮想環境の設定

  - Anaconda Promptで以下を行う

    ```sh
    > conda  env create -f setup.yml
    ```

  - 仮想環境の起動

    ```
    > conda activate personality
    ```