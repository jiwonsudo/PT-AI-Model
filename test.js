let model;

        async function loadModel() {
            model = await tf.loadLayersModel('tfjs_model/model.json');
            console.log("모델이 로드되었습니다!");
        }

        document.getElementById('proteinForm').addEventListener('submit', async function(event) {
            event.preventDefault();

            const height = parseFloat(document.getElementById('height').value);
            const weight = parseFloat(document.getElementById('weight').value);
            const age = parseFloat(document.getElementById('age').value);
            const activity_level = parseFloat(document.getElementById('activity_level').value);
            const goal = parseFloat(document.getElementById('goal').value);

            // 입력 데이터 배열로 변환
            const inputTensor = tf.tensor2d([[height, weight, age, activity_level, goal]], [1, 5]);

            // 모델 예측 실행
            const prediction = model.predict(inputTensor);
            const result = await prediction.data();
            
            // 결과 출력 (0에 가까우면 WPC, 1에 가까우면 WPI)
            document.getElementById('result').innerText = result[0] > 0.5 ? "WPI 추천" : "WPC 추천";
        });

        loadModel();