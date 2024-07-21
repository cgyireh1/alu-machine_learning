import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class PredictionPage extends StatefulWidget {
  @override
  _PredictionPageState createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _costOfLivingController = TextEditingController();
  final _rentIndexController = TextEditingController();
  final _groceriesIndexController = TextEditingController();
  final _restaurantPriceIndexController = TextEditingController();
  final _localPurchasingPowerIndexController = TextEditingController();

  double _predictedValue = 0.0;
  String _errorMessage = '';

  Future<void> _predictValue() async {
    try {
      final costOfLiving = double.parse(_costOfLivingController.text);
      final rentIndex = double.parse(_rentIndexController.text);
      final groceriesIndex = double.parse(_groceriesIndexController.text);
      final restaurantPriceIndex =
          double.parse(_restaurantPriceIndexController.text);
      final localPurchasingPowerIndex =
          double.parse(_localPurchasingPowerIndexController.text);

      final response = await http.post(
        Uri.parse('http://localhost:5000/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'cost_of_living_index': costOfLiving,
          'rent_index': rentIndex,
          'groceries_index': groceriesIndex,
          'restaurant_price_index': restaurantPriceIndex,
          'local_purchasing_power_index': localPurchasingPowerIndex,
        }),
      );

      if (response.statusCode == 200) {
        final predictedValue = jsonDecode(response.body)['predicted_value'];
        setState(() {
          _predictedValue = predictedValue;
          _errorMessage = '';
        });
      } else {
        setState(() {
          _predictedValue = 0.0;
          _errorMessage = 'Error: ${response.statusCode} - ${response.body}';
        });
      }
    } catch (e) {
      setState(() {
        _predictedValue = 0.0;
        _errorMessage = 'Error: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Prediction', style: TextStyle(color: Colors.white)),
        centerTitle: true,
        backgroundColor: Colors.pink,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _costOfLivingController,
              decoration:
                  const InputDecoration(labelText: 'Cost of Living Index'),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 16.0),
            TextField(
              controller: _rentIndexController,
              decoration: const InputDecoration(labelText: 'Rent Index'),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 16.0),
            TextField(
              controller: _groceriesIndexController,
              decoration: const InputDecoration(labelText: 'Groceries Index'),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 16.0),
            TextField(
              controller: _restaurantPriceIndexController,
              decoration:
                  const InputDecoration(labelText: 'Restaurant Price Index'),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 16.0),
            TextField(
              controller: _localPurchasingPowerIndexController,
              decoration: const InputDecoration(
                  labelText: 'Local Purchasing Power Index'),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 16.0),
            ElevatedButton(
              onPressed: _predictValue,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.pink,
                foregroundColor: Colors.white,
              ),
              child: const Text('Predict',
                  style:
                      TextStyle(fontSize: 18.0, fontWeight: FontWeight.bold)),
            ),
            const SizedBox(height: 16.0),
            if (_errorMessage.isNotEmpty)
              Text(
                _errorMessage,
                style: const TextStyle(color: Colors.pink),
              )
            else
              Text(
                'Predicted Value: $_predictedValue',
                style: const TextStyle(
                    fontSize: 18.0, fontWeight: FontWeight.bold),
              ),
          ],
        ),
      ),
    );
  }
}
