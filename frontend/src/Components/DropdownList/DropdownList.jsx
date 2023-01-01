import React from 'react';
import classes from './DropdownList.scss';
import { BsFillArrowDownCircleFill } from 'react-icons/bs';
class DropdownList extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            items: this.props.items ?? [],
            currentItem: 0,
        };
    }

    componentDidUpdate = () => {
        const {items, currentItem} = this.state
        if(items[currentItem] !== this.props.items[currentItem]){
            this.setState({currentItem: 0})
        }
    }

    onClickHandler = event => {
        const { items } = this.state;
        const currentItem = items.indexOf(event.target.textContent);
        this.props.setValue(items[currentItem]);
        this.setState({ currentItem });
    };

    render() {
        const { currentItem } = this.state;
        const { className, items } = this.props;
        return (
            <div className={`${classes.wrapper} ${className ?? ''}`}>
                <div className={classes.currentItem}>
                    {items[currentItem]} <BsFillArrowDownCircleFill />
                </div>
                <div className={classes.items}>
                    {items?.map((element, index) => (
                        <span
                            id={element + index}
                            key={element + index}
                            onClick={this.onClickHandler}>
                            {element}
                        </span>
                    ))}
                </div>
            </div>
        );
    }
}

export default DropdownList;
