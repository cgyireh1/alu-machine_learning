// ignore_for_file: prefer_final_fields, use_key_in_widget_constructors, library_private_types_in_public_api

import 'dart:convert';
import 'dart:math';
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
      final response = await http.post(
        Uri.parse(
            'https://ml-summative-heroku-24-78c86336b30f.herokuapp.com/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'cost_of_living_index': double.parse(_costOfLivingController.text),
          'rent_index': double.parse(_rentIndexController.text),
          'groceries_index': double.parse(_groceriesIndexController.text),
          'restaurant_price_index':
              double.parse(_restaurantPriceIndexController.text),
          'local_purchasing_power_index':
              double.parse(_localPurchasingPowerIndexController.text)
        }),
      );

      if (response.statusCode == 200) {
        final predictedValue =
            double.parse(jsonDecode(response.body)['predicted_value']);
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
        title: const Text('Cost Of Living Prediction App 😬',
            style: TextStyle(
                color: Colors.white,
                fontSize: 20.0,
                fontWeight: FontWeight.bold,
                fontFamily: 'Lobster')),
        centerTitle: true,
        backgroundColor: Colors.pink[300],
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(
            bottom: Radius.circular(20.0),
          ),
        ),
      ),
      body: Stack(
        children: [
          Container(
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  Color.fromARGB(255, 205, 209, 226),
                  Color.fromARGB(255, 110, 106, 87),
                ],
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.grey.withOpacity(0.5),
                  spreadRadius: 5,
                  blurRadius: 7,
                  offset: const Offset(0, 3),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Expanded(
                  child: SingleChildScrollView(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        _buildTextField(
                            _costOfLivingController, 'Cost of Living Index'),
                        const SizedBox(height: 16.0),
                        _buildTextField(_rentIndexController, 'Rent Index'),
                        const SizedBox(height: 16.0),
                        _buildTextField(
                            _groceriesIndexController, 'Groceries Index'),
                        const SizedBox(height: 16.0),
                        _buildTextField(_restaurantPriceIndexController,
                            'Restaurant Price Index'),
                        const SizedBox(height: 16.0),
                        _buildTextField(_localPurchasingPowerIndexController,
                            'Local Purchasing Power Index'),
                      ],
                    ),
                  ),
                ),
                ElevatedButton(
                  onPressed: _predictValue,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.pink[400],
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10.0),
                    ),
                    minimumSize: const Size(
                        200, 55), // Adjust the width and height as needed
                  ),
                  child: const Text(
                    'PREDICT',
                    style: TextStyle(
                      fontSize: 15.0, // Adjust the font size as needed
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                const SizedBox(height: 16.0),
                if (_errorMessage.isNotEmpty)
                  Text(
                    _errorMessage,
                    style: const TextStyle(color: Colors.pink),
                  )
                else
                  Text(
                    'Cost of Living Plus Rent Index: $_predictedValue',
                    style: const TextStyle(
                        fontSize: 21.0,
                        fontWeight: FontWeight.bold,
                        color: Colors.black54),
                  ),
              ],
            ),
          ),
          Positioned(
            top: 10,
            left: 10,
            child: CustomPaint(
              size: const Size(80, 560),
              painter: ShapePainter(
                color: Colors.pink.withOpacity(0.1),
                shape: ShapeType.rectangle,
              ),
            ),
          ),
          Positioned(
            bottom: 50,
            right: 40,
            child: CustomPaint(
              size: const Size(100, 100),
              painter: ShapePainter(
                color: Colors.pink.withOpacity(0.2),
                shape: ShapeType.circle,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTextField(TextEditingController controller, String labelText) {
    return TextField(
      controller: controller,
      decoration: InputDecoration(
        labelText: labelText,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10.0),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10.0),
          borderSide: const BorderSide(color: Colors.pink, width: 2.0),
        ),
      ),
      keyboardType: TextInputType.number,
    );
  }
}

class ShapePainter extends CustomPainter {
  final Color color;
  final ShapeType shape;

  ShapePainter({required this.color, required this.shape});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    switch (shape) {
      case ShapeType.circle:
        canvas.drawCircle(Offset(size.width, size.height),
            min(size.width, size.height), paint);
        break;
      case ShapeType.rectangle:
        canvas.drawRect(Offset.zero & size, paint);
        break;
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}

enum ShapeType { circle, rectangle }
