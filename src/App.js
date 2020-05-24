import React, { Component } from 'react';
import '@vkontakte/vkui/dist/vkui.css';
import { View, Panel, PanelHeader, FormLayout, Button, Input, CardGrid, Card, InfoRow, Progress, Div } from '@vkontakte/vkui';//пакеты из вк
// import Icon24DismissSubstract from '@vkontakte/icons/dist/24/dismiss_substract';//npm i @vkontakte/icons
import Icon28BrainOutline from '@vkontakte/icons/dist/28/brain_outline';
// import Icon28WriteOutline from '@vkontakte/icons/dist/28/write_outline';
// import './App.css'
import * as tf from '@tensorflow/tfjs';//npm install @tensorflow/tfjs

class App extends Component {
	constructor(props) {
		super(props);
		this.state = {
			tp: '',
			result: [],
			district: ['нулевой', 'ЦРЭС', 'ЮРЭС', 'ЗРЭС', 'СРЭС', 'ВРЭС', 'ЮВРЭС']
		}
	}

	componentDidMount() {
		//вызываем предыдущее состояние из локалсториджа
		const lastState = localStorage.district
		if (lastState) {
			// console.log(lastState)
			this.setState(JSON.parse(lastState))
		}
	}

	//обязательно используем стрелочные фунции чтоб не прописывать методы в конструкторе
	tpChange = (event) => {
		this.setState({ tp: event.target.value });
	}

	//вычисляем из названия принадлежность РЭС
	predictDist = async () => {
		/**функция для преобразования названия тп в массив чисел перед входом в нейронный предсказатель
        @param {string} str наименование электроустановки
        @return {[number]} массив чисел.
        */
		function padSequence(str) {
			str = str.toLowerCase()
			const re = /-/gi;
			str = str.replace(re, '');//удаляем тире
			str = str.replace(/  +/g, '');//удаляем пробелы
			str = str.replace('рп', '1');
			str = str.replace('тп', '2');
			str = str.replace('пп', '3');
			str = str.replace(/[A-z]+/g, '2')//все латинские кодируются в ТП
			let seq = []
			let counter = 5
			for (let i of str) {
				if (counter > 0) {
					seq.push(Number(i))
					counter -= 1
				}
			}
			const len_seq = 5
			if (seq.length < len_seq) {
				let zerro = len_seq - seq.length
				console.log('нехватает нулей ' + zerro)
				while (zerro) {
					zerro -= 1
					seq.push(0)//добавляем нули справа
				}
			}
			console.log(`после очистки и преобразования ${seq}`)
			return seq
		}

		let str = this.state.tp
		let num_seq = padSequence(str)
		console.log('model begin load:)))))')
		// const model1 = await tf.loadLayersModel('https://cors-anywhere.herokuapp.com/https://ilgiz.h1n.ru/district_rp_tp/model/model.json')//это для разработки локальная модель не загрузится
		const model1 = await tf.loadLayersModel('./model/model.json')//обязательно нужно помещать папку с моделями в /public
		console.log('model loaded:)))))')
		model1.summary()
		// model1.predict(tf.tensor2d([0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 97, 3, 76, 75, 2, 113], [1, 16])).print()
		// model1.predict(tf.tensor2d(num_seq, [1, 5])).print()
		let pred = await model1.predict(tf.tensor2d(num_seq, [1, 5])).flatten().array()
		console.log(pred)
		this.setState({ result: pred })
		localStorage.district = JSON.stringify(this.state);//сохраняем стейт в локалсторадже
	}
	render() {
		return (
			<View id="view" activePanel="panel">
				<Panel id="panel">
					<PanelHeader>какой РЭС?</PanelHeader>
					<FormLayout align="center" >
						<Input placeholder="тп-2222" top="введите название элекроустановки" align="center" value={this.state.tp} onChange={this.tpChange} />
						<Button onClick={this.predictDist} before={<Icon28BrainOutline />} size="l">кому принадлежит?</Button>
						{this.state.result.length ?
							<CardGrid>
								<Card size="l" mode="outline">
									{this.state.result.map(
										(element, index) => ((index > 0) && (Math.round(element * 100) > 0)) ?
											<Div key={index}>
												<InfoRow header={`${this.state.district[index]} уверенность ${Math.round(element * 100)}%`}>
													<Progress value={Math.round(element * 100)} />
												</InfoRow>
											</Div> : null)}
								</Card>
							</CardGrid> : null}
					</FormLayout>
				</Panel>
			</View>
		);
	}
}

export default App;

