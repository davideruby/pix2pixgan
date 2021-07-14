docker build -t unitopatho-generative .
docker tag unitopatho-generative daviderubinetti/unitopatho-generative
docker push daviderubinetti/unitopatho-generative
kubectl apply -f pod-train.yaml
echo "Done"