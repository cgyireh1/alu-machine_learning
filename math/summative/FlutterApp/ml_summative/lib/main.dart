import 'package:flutter/material.dart';
import 'prediction_page.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Prediction App',
      home: PredictionPage(),
    );
  }
}
